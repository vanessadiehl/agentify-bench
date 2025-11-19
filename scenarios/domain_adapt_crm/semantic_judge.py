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


def _extract_gold_entities(episode: dict) -> set[tuple[str, str]]:
    """
    Extract gold entities as (type, id_or_name) pairs from YAML.
    """
    result: set[tuple[str, str]] = set()
    gold = (episode.get("gold_ontology") or {}).get("entities") or []
    for e in gold:
        etype = (e.get("type") or "").strip().lower()
        name = ""
        if etype == "account":
            name = (e.get("name") or "").strip()
        elif etype == "case":
            name = (e.get("subject") or "").strip()
        else:
            name = (e.get("name") or e.get("subject") or "").strip()

        if etype and name:
            result.add((etype, name.lower()))
    return result


def _compute_prf(gold: set[tuple[str, str]], pred: set[tuple[str, str]]) -> dict:
    """Compute precision / recall / F1 for entity matching."""
    if not gold and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

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
    """Green agent that evaluates CRM semantic mapping accuracy."""
    
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

            # 1) Load the first episode spec
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
                    f"Loaded episode id={ep_id}, title={ep_title}. Calling CRM mapper next..."
                ),
            )

            # 2) Build prompt for the crm_mapper participant
            crm_agent_url = req.participants.get("crm_mapper")
            if not crm_agent_url:
                msg = "Missing participant 'crm_mapper' in request.participants."
                logger.error(msg)
                await updater.update_status(
                    TaskState.errored, new_agent_text_message(msg)
                )
                return

            user_message = (episode.get("task") or {}).get("user_message", "")
            expected_format = (episode.get("requirements") or {}).get(
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

            logger.info(
                f"[Semantic] Sending prompt to crm_mapper at {crm_agent_url}"
            )
            agent_response = await self._tool_provider.talk_to_agent(
                prompt,
                str(crm_agent_url),
                new_conversation=True,
            )

            logger.info(
                f"[Semantic] Raw response from crm_mapper:\n{agent_response}"
            )

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Received response from crm_mapper (first 500 chars):\n{agent_response[:500]}"
                ),
            )

            # 3) Try to parse the agent response as JSON and compute entity metrics
            metrics = {}
            pred_entities: set[tuple[str, str]] = set()
            gold_entities: set[tuple[str, str]] = set()

            try:
                cleaned = _clean_json_block(agent_response)
                logger.info(f"[Semantic] Cleaned agent JSON string:\n{cleaned[:500]}")
                agent_json = json.loads(cleaned)
                pred_entities = _extract_pred_entities(agent_json)
                gold_entities = _extract_gold_entities(episode)
                metrics = _compute_prf(gold_entities, pred_entities)
                logger.info(
                    f"[Semantic] Entity metrics: {metrics} | gold={gold_entities} | pred={pred_entities}"
                )
            except Exception as e:
                logger.exception(
                    f"[Semantic] Failed to parse agent JSON or compute metrics: {e}"
                )
                metrics = {
                    "error": str(e),
                    "parsed_ok": False,
                }
            else:
                metrics["parsed_ok"] = True

            # 4) Build EvalResult with metrics and a preview of the response
            result = EvalResult(
                winner="n/a",
                detail={
                    "episode_id": ep_id,
                    "episode_title": ep_title,
                    "note": "Semantic benchmark; entity-level metrics computed from CRM agent response.",
                    "agent_raw_response_preview": agent_response[:500],
                    "gold_entities": list(sorted(gold_entities)),
                    "pred_entities": list(sorted(pred_entities)),
                    "entity_metrics": metrics,
                },
            )

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=result.model_dump_json())),
                ],
                name="Result",
            )

            await updater.update_status(
                TaskState.completed,
                new_agent_text_message(
                    f"Semantic evaluation completed for episode {ep_id}."
                ),
            )
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