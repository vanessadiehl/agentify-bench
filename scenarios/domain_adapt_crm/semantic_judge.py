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
from agentbeats.server.tasks import TaskUpdater

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

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        try:
            episode_path = req.config["episodes"][0]
            with open(episode_path, "r") as f:
                episode = yaml.safe_load(f)

            crm_url = req.participants["crm_mapper"]
            turns = episode.get("turns", [])
            all_turn_scores = []
            
            # STATE TRACKING FOR CONSISTENCY
            prev_turn_correct_rels = set()
            prev_agent_json = None

            for i, turn_data in enumerate(turns):
                user_msg = turn_data["user_message"]
                
                # Context injection: Give the agent its previous state to maintain
                context = f"\n\nCurrent CRM State:\n{json.dumps(prev_agent_json)}" if prev_agent_json else ""
                prompt = f"{user_msg}{context}\n\nOutput strictly in JSON."

                agent_raw = await self._tool_provider.talk_to_agent(prompt, str(crm_url))
                
                turn_report = {"turn": i + 1, "parsed_ok": True}
                
                try:
                    cleaned = _clean_json_block(agent_raw)
                    agent_json = json.loads(cleaned)
                    prev_agent_json = agent_json # Update state
                    
                    # 1. Score Entities
                    gold_ents = _extract_gold_entities(turn_data)
                    pred_ents = _extract_pred_entities(agent_json)
                    turn_report["entity_metrics"] = _compute_metrics(gold_ents, pred_ents)
                    
                    # 2. Score Relationships
                    gold_rels = _extract_relationships(turn_data, is_gold=True)
                    pred_rels = _extract_relationships(agent_json, is_gold=False)
                    rel_metrics = _compute_metrics(gold_rels, pred_rels)
                    turn_report["rel_metrics"] = rel_metrics
                    
                    # 3. Score "Agent Persistence" (Real Consistency)
                    # How many of the correct relationships from the LAST turn did the agent keep?
                    if i > 0 and prev_turn_correct_rels:
                        maintained = len(prev_turn_correct_rels & pred_rels)
                        # We only penalize for what was correct and SHOULD still be there
                        # (Simplification: assumes gold doesn't explicitly delete a previous correct rel)
                        turn_report["persistence"] = maintained / len(prev_turn_correct_rels)
                    else:
                        turn_report["persistence"] = 1.0

                    # Update what was correct this turn for the next turn's check
                    prev_turn_correct_rels = gold_rels & pred_rels

                except json.JSONDecodeError:
                    turn_report["parsed_ok"] = False
                    turn_report["error"] = "JSON_DECODE_FAILURE"
                    prev_agent_json = None 

                all_turn_scores.append(turn_report)

            # Final Aggregation
            avg_ent_f1 = sum(t.get("entity_metrics", {}).get("f1", 0) for t in all_turn_scores) / len(turns)
            avg_rel_f1 = sum(t.get("rel_metrics", {}).get("f1", 0) for t in all_turn_scores) / len(turns)
            
            result = EvalResult(
                winner="n/a",
                detail={
                    "metrics": {"avg_ent_f1": avg_ent_f1, "avg_rel_f1": avg_rel_f1},
                    "turns": all_turn_scores
                }
            )

            await updater.add_artifact([Part(root=TextPart(text=result.model_dump_json()))], name="Detailed_Scores")
            await updater.update_status(TaskState.completed, new_agent_text_message("Evaluation Finished."))

        finally:
            self._tool_provider.reset()

# ... (rest of the A2A boilerplate remains same)