import argparse
import contextlib
import uvicorn
import asyncio
import logging
import yaml
import json
from dotenv import load_dotenv

load_dotenv()

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider

from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_agent_text_message
from a2a.server.tasks import TaskUpdater

# Note: Assuming semantic_judge_common is in your path
from semantic_judge_common import SemanticEvalResult, semantic_judge_agent_card

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic_judge")

def _normalize(text: str) -> str:
    """Strict structural normalization only. No semantic guessing."""
    return str(text).strip().lower()

def _clean_json_block(raw: str) -> str:
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
    result: set[tuple[str, str]] = set()
    entities = agent_json.get("entities") or []
    for e in entities:
        etype = _normalize(e.get("entity_type") or "")
        # Prioritize name/subject as the 'key' for the entity
        ename = _normalize(e.get("name") or e.get("subject") or e.get("lot_number") or "")
        if etype and ename:
            result.add((etype, ename))
    return result

def _extract_gold_entities(turn_data: dict) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    gold = (turn_data.get("gold_ontology") or {}).get("entities") or []
    for e in gold:
        etype = _normalize(e.get("type") or "")
        ename = _normalize(e.get("name") or e.get("subject") or e.get("lot_number") or e.get("description") or "")
        if etype and ename:
            result.add((etype, ename))
    return result

def _extract_relationships(data_source: dict, is_gold: bool = False) -> set[tuple[str, str, str]]:
    """Generic extractor for both gold and predicted relationships."""
    if is_gold:
        rels = (data_source.get("gold_ontology") or {}).get("relationships") or []
    else:
        rels = data_source.get("relationships") or []
        
    result: set[tuple[str, str, str]] = set()
    for r in rels:
        f = _normalize(r.get("from") or "")
        t = _normalize(r.get("type") or "")
        to = _normalize(r.get("to") or "")
        if f and t and to:
            result.add((f, t, to))
    return result

def _compute_metrics(gold: set, pred: set):
    tp = gold & pred
    fp = pred - gold
    fn = gold - pred
    
    precision = len(tp) / len(pred) if pred else 1.0 if not gold else 0.0
    recall = len(tp) / len(gold) if gold else 1.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "f1": f1, "precision": precision, "recall": recall,
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "tp_list": list(tp), "fp_list": list(fp), "fn_list": list(fn)
    }

class SemanticJudge(GreenAgent):
    def __init__(self):
        self._required_roles = ["crm_mapper"]
        self._required_config_keys = ["episodes"]
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """Validate that the request has required roles and config."""
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        
        episodes = request.config.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            return False, "Config 'episodes' must be a non-empty list of file paths."
        
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        try:
            episode_paths = req.config["episodes"]  # Now handles MULTIPLE episodes
            if not episode_paths:
                raise ValueError("No episodes provided in config")
            
            crm_url = req.participants["crm_mapper"]
            
            # Storage for all episodes
            all_episodes_results = []
            all_entity_f1_scores = []
            all_rel_f1_scores = []
            all_persistence_scores = []

            # LOOP through each episode
            for episode_idx, episode_path in enumerate(episode_paths):
                logger.info(f"[Semantic] Processing episode {episode_idx + 1}/{len(episode_paths)}: {episode_path}")
                
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Processing episode {episode_idx + 1}/{len(episode_paths)}: {episode_path}")
                )
                
                with open(episode_path, "r") as f:
                    episode = yaml.safe_load(f)

                ep_id = episode.get("id", f"episode_{episode_idx}")
                ep_title = episode.get("title", f"Episode {episode_idx + 1}")
                turns = episode.get("turns", [])
                
                if not turns:
                    logger.warning(f"[Semantic] Episode {ep_id} has no turns")
                    continue

                all_turn_scores = []
                
                # STATE TRACKING FOR CONSISTENCY (per episode)
                prev_turn_correct_rels = set()
                prev_agent_json = None

                for turn_idx, turn_data in enumerate(turns):
                    user_msg = turn_data["user_message"]
                    
                    # Context injection: Give the agent its previous state to maintain
                    context = f"\n\nCurrent CRM State:\n{json.dumps(prev_agent_json)}" if prev_agent_json else ""
                    prompt = f"{user_msg}{context}\n\nOutput strictly in JSON."

                    agent_raw = await self._tool_provider.talk_to_agent(prompt, str(crm_url), new_conversation=True)
                    
                    turn_report = {"turn": turn_idx + 1, "parsed_ok": True}
                    
                    try:
                        cleaned = _clean_json_block(agent_raw)
                        agent_json = json.loads(cleaned)
                        prev_agent_json = agent_json  # Update state
                        
                        # 1. Score Entities
                        gold_ents = _extract_gold_entities(turn_data)
                        pred_ents = _extract_pred_entities(agent_json)
                        entity_metrics = _compute_metrics(gold_ents, pred_ents)
                        turn_report["entity_metrics"] = entity_metrics
                        
                        # 2. Score Relationships
                        gold_rels = _extract_relationships(turn_data, is_gold=True)
                        pred_rels = _extract_relationships(agent_json, is_gold=False)
                        rel_metrics = _compute_metrics(gold_rels, pred_rels)
                        turn_report["rel_metrics"] = rel_metrics
                        
                        # 3. Score "Agent Persistence" (Real Consistency)
                        # How many of the correct relationships from the LAST turn did the agent keep?
                        if turn_idx > 0 and prev_turn_correct_rels:
                            maintained = len(prev_turn_correct_rels & pred_rels)
                            turn_report["persistence"] = maintained / len(prev_turn_correct_rels)
                        else:
                            turn_report["persistence"] = 1.0

                        # Update what was correct this turn for the next turn's check
                        prev_turn_correct_rels = gold_rels & pred_rels

                        logger.info(
                            f"[Semantic] Episode {ep_id}, Turn {turn_idx + 1}: "
                            f"Entity F1={entity_metrics['f1']:.3f}, "
                            f"Rel F1={rel_metrics['f1']:.3f}, "
                            f"Persistence={turn_report['persistence']:.3f}"
                        )

                    except json.JSONDecodeError as e:
                        logger.error(f"[Semantic] Episode {ep_id}, Turn {turn_idx + 1}: JSON decode failed: {e}")
                        turn_report["parsed_ok"] = False
                        turn_report["error"] = "JSON_DECODE_FAILURE"
                        prev_agent_json = None 

                    all_turn_scores.append(turn_report)

                # Episode-level aggregation
                episode_entity_f1s = [t.get("entity_metrics", {}).get("f1", 0) for t in all_turn_scores if t.get("parsed_ok")]
                episode_rel_f1s = [t.get("rel_metrics", {}).get("f1", 0) for t in all_turn_scores if t.get("parsed_ok")]
                episode_persistence = [t.get("persistence", 0) for t in all_turn_scores if t.get("parsed_ok")]
                
                episode_avg_ent_f1 = sum(episode_entity_f1s) / len(episode_entity_f1s) if episode_entity_f1s else 0.0
                episode_avg_rel_f1 = sum(episode_rel_f1s) / len(episode_rel_f1s) if episode_rel_f1s else 0.0
                episode_avg_persistence = sum(episode_persistence) / len(episode_persistence) if episode_persistence else 0.0
                
                episode_result = {
                    "episode_id": ep_id,
                    "episode_title": ep_title,
                    "avg_entity_f1": round(episode_avg_ent_f1, 3),
                    "avg_rel_f1": round(episode_avg_rel_f1, 3),
                    "avg_persistence": round(episode_avg_persistence, 3),
                    "turns": all_turn_scores
                }
                all_episodes_results.append(episode_result)
                
                # Accumulate for global metrics
                all_entity_f1_scores.extend(episode_entity_f1s)
                all_rel_f1_scores.extend(episode_rel_f1s)
                all_persistence_scores.extend(episode_persistence)
                
                logger.info(f"[Semantic] Episode {ep_id} completed: Entity F1={episode_avg_ent_f1:.3f}, Rel F1={episode_avg_rel_f1:.3f}")

            # GLOBAL aggregation across ALL episodes
            global_avg_ent_f1 = sum(all_entity_f1_scores) / len(all_entity_f1_scores) if all_entity_f1_scores else 0.0
            global_avg_rel_f1 = sum(all_rel_f1_scores) / len(all_rel_f1_scores) if all_rel_f1_scores else 0.0
            global_avg_persistence = sum(all_persistence_scores) / len(all_persistence_scores) if all_persistence_scores else 0.0

            result = EvalResult(
                winner="n/a",
                detail={
                    "global_metrics": {
                        "avg_entity_f1": round(global_avg_ent_f1, 3),
                        "avg_rel_f1": round(global_avg_rel_f1, 3),
                        "avg_persistence": round(global_avg_persistence, 3),
                        "num_episodes": len(all_episodes_results),
                        "total_turns": sum(len(e["turns"]) for e in all_episodes_results)
                    },
                    "episodes": all_episodes_results
                }
            )

            await updater.add_artifact([Part(root=TextPart(text=result.model_dump_json()))], name="Detailed_Scores")
            
            status_msg = (
                f"Evaluation completed across {len(all_episodes_results)} episodes. "
                f"Global Entity F1: {global_avg_ent_f1:.3f}, "
                f"Global Rel F1: {global_avg_rel_f1:.3f}, "
                f"Global Persistence: {global_avg_persistence:.3f}"
            )
            await updater.update_status(TaskState.completed, new_agent_text_message(status_msg))

            logger.info(f"[Semantic] Evaluation finished. {status_msg}")

        except Exception as e:
            logger.exception(f"[Semantic] Evaluation failed: {e}")
            await updater.update_status(TaskState.errored, new_agent_text_message(f"Evaluation failed: {str(e)}"))
        finally:
            self._tool_provider.reset()


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A semantic judge agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument(
        "--cloudflare-quick-tunnel",
        action="store_true",
        help="Use a Cloudflare quick tunnel. Requires cloudflared.",
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

        from a2a.server.apps import A2AStarletteApplication
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.server.tasks import InMemoryTaskStore

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