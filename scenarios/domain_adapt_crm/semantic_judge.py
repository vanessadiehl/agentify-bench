import argparse
import contextlib
import uvicorn
import asyncio
import logging
import yaml
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal


load_dotenv()

from google import genai
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    Part,
    TextPart,
)
from a2a.utils import new_agent_text_message

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider

from semantic_judge_common import SemanticEvalResult, semantic_judge_agent_card


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_judge")


def _clean_json_block(raw: str) -> str:
    """Strip ```json fences if the agent wrapped the JSON in a code block."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_pred_entities(agent_json: dict) -> set[tuple[str, str]]:
    """
    Extract predicted entities as (type, id_or_name) pairs from agent JSON.
    """
    result: set[tuple[str, str]] = set()
    entities = agent_json.get("entities") or []
    for e in entities:
        etype = (e.get("entity_type") or "").strip().lower()
        eid = (e.get("entity_id") or "").strip()

        if not eid:
            props = e.get("properties") or {}
            eid = (props.get("name") or "").strip()

        if not eid:
            eid = (e.get("name") or "").strip()

        if not eid and etype == "case":
            eid = (e.get("subject") or "").strip()

        if etype and eid:
            result.add((etype, eid.lower()))

    return result


def _extract_gold_entities(turn_data: dict) -> set[tuple[str, str]]:
    """
    Extract gold entities as (type, id_or_name) pairs from YAML turn.
    """
    result: set[tuple[str, str]] = set()
    gold = (turn_data.get("gold_ontology") or {}).get("entities") or []
    for e in gold:
        etype = (e.get("type") or "").strip().lower()
        name = ""
        if etype == "account":
            name = (e.get("name") or "").strip()
        elif etype == "case":
            name = (e.get("subject") or "").strip()
        elif etype == "contact":
            name = (e.get("name") or "").strip()
        elif etype == "property":
            name = (e.get("lot_number") or "").strip()
        elif etype == "interaction":
            name = (e.get("description") or "").strip()
        else:
            name = (e.get("name") or e.get("subject") or "").strip()

        if etype and name:
            result.add((etype, name.lower()))
    return result


def _extract_pred_relationships(agent_json: dict) -> set[tuple[str, str, str]]:
    """
    Extract predicted relationships as (from, type, to) tuples from agent JSON.
    """
    result: set[tuple[str, str, str]] = set()
    relationships = agent_json.get("relationships") or []
    for r in relationships:
        from_entity = (r.get("from") or "").strip().lower()
        rel_type = (r.get("type") or "").strip().lower()
        to_entity = (r.get("to") or "").strip().lower()
        
        if from_entity and rel_type and to_entity:
            result.add((from_entity, rel_type, to_entity))
    
    return result


def _extract_gold_relationships(turn_data: dict) -> set[tuple[str, str, str]]:
    """
    Extract gold relationships as (from, type, to) tuples from YAML turn.
    """
    result: set[tuple[str, str, str]] = set()
    gold = (turn_data.get("gold_ontology") or {}).get("relationships") or []
    for r in gold:
        from_entity = (r.get("from") or "").strip().lower()
        rel_type = (r.get("type") or "").strip().lower()
        to_entity = (r.get("to") or "").strip().lower()
        
        if from_entity and rel_type and to_entity:
            result.add((from_entity, rel_type, to_entity))
    
    return result


def _compute_prf(gold: set[tuple], pred: set[tuple]) -> dict:
    """Compute precision / recall / F1 for matching."""
    if not gold and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


class SemanticJudge(GreenAgent):
    """Green agent that evaluates CRM semantic mapping accuracy across multiple turns."""
    
    def __init__(self):
        self._required_roles = ["crm_mapper"]
        self._required_config_keys = ["episodes"]
        self._client = genai.Client()
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(
            request.config.keys()
        )
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        episodes = request.config.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            return False, "Config 'episodes' must be a non-empty list of file paths."
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(
            f"Starting semantic CRM evaluation. Request config: {req.config}"
        )

        try:
            episodes = req.config.get("episodes", [])
            if not episodes:
                msg = "No episodes provided in config."
                logger.error(msg)
                await updater.update_status(
                    TaskState.errored, new_agent_text_message(msg)
                )
                return

            # Load episode
            episode_path = episodes[0]
            logger.info(f"[Semantic] Loading episode spec from: {episode_path}")

            with open(episode_path, "r", encoding="utf-8") as f:
                episode = yaml.safe_load(f)

            ep_id = episode.get("id")
            ep_title = episode.get("title")
            logger.info(f"[Semantic] Episode loaded: id={ep_id}, title={ep_title}")

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Loaded episode id={ep_id}, title={ep_title}. Calling CRM mapper..."
                ),
            )

            # Get crm_mapper URL
            crm_agent_url = req.participants.get("crm_mapper")
            if not crm_agent_url:
                msg = "Missing participant 'crm_mapper' in request.participants."
                logger.error(msg)
                await updater.update_status(
                    TaskState.errored, new_agent_text_message(msg)
                )
                return

            # Get all turns
            turns = episode.get("turns", [])
            if not turns:
                msg = "Episode has no turns defined."
                logger.error(msg)
                await updater.update_status(
                    TaskState.errored, new_agent_text_message(msg)
                )
                return

            all_turn_scores = []
            prev_gold_relationships = set()

            # LOOP through each turn
            for turn_num, turn_data in enumerate(turns):
                logger.info(f"[Semantic] Processing turn {turn_num + 1}/{len(turns)}")

                # Build prompt for this turn
                user_message = turn_data.get("user_message", "")
                expected_format = (turn_data.get("requirements") or {}).get(
                    "expected_output_format", ""
                )

                prompt = f"""
You are a CRM semantic mapping agent.

Task:
{user_message}

Goal:
Map the above domain-specific description into a canonical CRM ontology.

Expected output format:
{expected_format}

Please follow the expected JSON format as closely as possible.
"""

                # Call agent for this turn
                logger.info(f"[Semantic] Sending Turn {turn_num + 1} prompt to crm_mapper")
                agent_response = await self._tool_provider.talk_to_agent(
                    prompt,
                    str(crm_agent_url),
                    new_conversation=True,
                )

                logger.info(f"[Semantic] Raw response from Turn {turn_num + 1}:\n{agent_response}")

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Received response from Turn {turn_num + 1} (first 300 chars):\n{agent_response[:300]}"
                    ),
                )

                # Parse JSON and extract entities/relationships
                try:
                    cleaned = _clean_json_block(agent_response)
                    agent_json = json.loads(cleaned)
                    
                    # Extract predicted entities and relationships
                    pred_entities = _extract_pred_entities(agent_json)
                    pred_relationships = _extract_pred_relationships(agent_json)
                    
                    # Extract gold entities and relationships for this turn
                    gold_entities = _extract_gold_entities(turn_data)
                    gold_relationships = _extract_gold_relationships(turn_data)
                    
                    # Compute metrics
                    entity_metrics = _compute_prf(gold_entities, pred_entities)
                    relationship_metrics = _compute_prf(gold_relationships, pred_relationships)
                    
                    # Compute consistency (how many old relationships are maintained?)
                    consistency = 1.0
                    if turn_num > 0:
                        maintained = len(prev_gold_relationships & gold_relationships)
                        if prev_gold_relationships:
                            consistency = maintained / len(prev_gold_relationships)
                        else:
                            consistency = 1.0  # If no previous relationships, perfect consistency
                    
                    # Format consistency string for logging
                    consistency_str = f"{consistency:.2f}" if turn_num > 0 else "N/A"
                    logger.info(
                        f"[Semantic] Turn {turn_num + 1} metrics: "
                        f"Entity F1={entity_metrics['f1']:.2f}, "
                        f"Relationship F1={relationship_metrics['f1']:.2f}, "
                        f"Consistency={consistency_str}"
                    )

                    # Store turn scores
                    turn_score = {
                        "turn": turn_num + 1,
                        "entity_f1": entity_metrics["f1"],
                        "relationship_f1": relationship_metrics["f1"],
                        "consistency": consistency if turn_num > 0 else None,
                        "entity_details": {
                            "precision": entity_metrics["precision"],
                            "recall": entity_metrics["recall"],
                            "tp": entity_metrics["tp"],
                            "fp": entity_metrics["fp"],
                            "fn": entity_metrics["fn"],
                        },
                        "relationship_details": {
                            "precision": relationship_metrics["precision"],
                            "recall": relationship_metrics["recall"],
                            "tp": relationship_metrics["tp"],
                            "fp": relationship_metrics["fp"],
                            "fn": relationship_metrics["fn"],
                        },
                    }
                    all_turn_scores.append(turn_score)
                    
                    # Update for next turn's consistency check
                    prev_gold_relationships = gold_relationships

                except Exception as e:
                    logger.exception(f"[Semantic] Failed to process Turn {turn_num + 1}: {e}")
                    all_turn_scores.append({
                        "turn": turn_num + 1,
                        "error": str(e),
                        "parsed_ok": False,
                    })

            # Compute aggregate metrics
            entity_f1_scores = [s["entity_f1"] for s in all_turn_scores if "entity_f1" in s]
            relationship_f1_scores = [s["relationship_f1"] for s in all_turn_scores if "relationship_f1" in s]
            consistency_scores = [s["consistency"] for s in all_turn_scores if s.get("consistency") is not None]

            avg_entity_f1 = sum(entity_f1_scores) / len(entity_f1_scores) if entity_f1_scores else 0.0
            avg_relationship_f1 = sum(relationship_f1_scores) / len(relationship_f1_scores) if relationship_f1_scores else 0.0
            avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else None

            # Format consistency for final message
            consistency_final = f"{avg_consistency:.3f}" if avg_consistency is not None else "N/A"

            # Build final result
            result = EvalResult(
                winner="n/a",
                detail={
                    "episode_id": ep_id,
                    "episode_title": ep_title,
                    "note": "Multi-turn semantic benchmark with entity, relationship, and consistency scoring.",
                    "per_turn_scores": all_turn_scores,
                    "aggregate_metrics": {
                        "avg_entity_f1": round(avg_entity_f1, 3),
                        "avg_relationship_f1": round(avg_relationship_f1, 3),
                        "avg_consistency": round(avg_consistency, 3) if avg_consistency is not None else None,
                        "num_turns": len(turns),
                    },
                },
            )

            # Add artifact FIRST with error handling
            try:
                await updater.add_artifact(
                    parts=[
                        Part(root=TextPart(text=result.model_dump_json())),
                    ],
                    name="Result",
                )
                logger.info(f"[Semantic] Artifact added successfully")
            except Exception as e:
                logger.error(f"[Semantic] Failed to add artifact: {e}")

            # Then update status with error handling
            try:
                await updater.update_status(
                    TaskState.completed,
                    new_agent_text_message(
                        f"Semantic evaluation completed for episode {ep_id} with {len(turns)} turns. "
                        f"Avg Entity F1: {avg_entity_f1:.3f}, "
                        f"Avg Relationship F1: {avg_relationship_f1:.3f}, "
                        f"Avg Consistency: {consistency_final}"
                    ),
                )
                logger.info(f"[Semantic] Status updated to completed")
            except Exception as e:
                logger.error(f"[Semantic] Failed to update status: {e}")

        finally:
            self._tool_provider.reset()


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A semantic judge agent.")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=9009, help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url", type=str, help="External URL to provide in the agent card"
    )
    parser.add_argument(
        "--cloudflare-quick-tunnel",
        action="store_true",
        help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url",
    )
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel

        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(
            args.card_url or f"http://{args.host}:{args.port}/"
        )

    async with agent_url_cm as agent_url:
        agent = SemanticJudge()
        executor = GreenExecutor(agent)
        agent_card = semantic_judge_agent_card("SemanticJudge", agent_url)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())