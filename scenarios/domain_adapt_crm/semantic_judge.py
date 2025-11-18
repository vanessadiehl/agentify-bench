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

# from debate_judge_common import DebateEval, debate_judge_agent_card
from semantic_judge_common import DebateEval, debate_judge_agent_card


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debate_judge")


def _clean_json_block(raw: str) -> str:
    """Strip ```json fences if the agent wrapped the JSON in a code block."""
    text = raw.strip()
    if text.startswith("```"):
        # Remove first fence line
        lines = text.splitlines()
        # Drop leading ``` or ```json line
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        # Drop trailing ``` if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_pred_entities(agent_json: dict) -> set[tuple[str, str]]:
    """
    Extract predicted entities as (type, id_or_name) pairs from agent JSON.

    Supports shapes like:
    {
      "entities": [
        {
          "entity_type": "Account",
          "entity_id": "ACME Construction",   # optional
          "name": "ACME Construction",        # optional
          "properties": { "name": "ACME Construction", ... }  # optional
        },
        ...
      ]
    }
    """
    result: set[tuple[str, str]] = set()
    entities = agent_json.get("entities") or []
    for e in entities:
        etype = (e.get("entity_type") or "").strip().lower()

        # Try several options for the identifier / name
        eid = (e.get("entity_id") or "").strip()

        # Fallback 1: properties.name
        if not eid:
            props = e.get("properties") or {}
            eid = (props.get("name") or "").strip()

        # Fallback 2: top-level "name"
        if not eid:
            eid = (e.get("name") or "").strip()

        # Fallback 3: for cases, use "subject"
        if not eid and etype == "case":
            eid = (e.get("subject") or "").strip()

        if etype and eid:
            result.add((etype, eid.lower()))

    return result


def _extract_gold_entities(episode: dict) -> set[tuple[str, str]]:
    """
    Extract gold entities as (type, id_or_name) pairs from YAML.

    For now:
      - Accounts use the `name` field
      - Cases use the `subject` field
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
            # fallback: try name, then subject
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


class DebateJudge(GreenAgent):
    def __init__(self):
        self._required_roles = ["crm_semantic_system"]
        self._required_config_keys = ["episodes"]
        # self._required_roles = ["pro_debater", "con_debater"]
        # self._required_config_keys = ["episodes"]
        # self._required_config_keys = ["topic", "num_rounds"]
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
        # try:
        # int(request.config["num_rounds"])
        # except Exception as e:
        # return False, f"Can't parse num_rounds: {e}"
        episodes = request.config.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            return False, "Config 'episodes' must be a non-empty list of file paths."
        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(
            f"Starting semantic CRM evaluation (M3.5 skeleton). Request config: {req.config}"
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
                    f"Loaded episode id={ep_id}, title={ep_title}. Calling CRM semantic system next..."
                ),
            )

            # 2) Build prompt for the crm_semantic_system participant
            crm_agent_url = req.participants.get("crm_semantic_system")
            if not crm_agent_url:
                msg = (
                    "Missing participant 'crm_semantic_system' in request.participants."
                )
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
                f"[Semantic] Sending prompt to crm_semantic_system at {crm_agent_url}"
            )
            agent_response = await self._tool_provider.talk_to_agent(
                prompt,
                str(crm_agent_url),
                new_conversation=True,
            )

            logger.info(
                f"[Semantic] Raw response from crm_semantic_system:\n{agent_response}"
            )

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Received response from crm_semantic_system (first 500 chars):\n{agent_response[:500]}"
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
                    f"Semantic evaluation skeleton completed for episode {ep_id} (CRM agent called)."
                ),
            )
        finally:
            self._tool_provider.reset()

    async def orchestrate_debate(
        self,
        participants: dict[str, str],
        topic: str,
        num_rounds: int,
        updater: TaskUpdater,
    ) -> dict[str, list[str]]:
        debate: dict[str, list[str]] = {"pro_debater": [], "con_debater": []}

        async def turn(role: str, prompt: str) -> str:
            response = await self._tool_provider.talk_to_agent(
                prompt, str(participants[role]), new_conversation=False
            )
            logger.info(f"{role}: {response}")
            debate[role].append(response)
            await updater.update_status(
                TaskState.working, new_agent_text_message(f"{role}: {response}")
            )
            return response

        # Opening turns
        response = await turn(
            "pro_debater", f"Debate Topic: {topic}. Present your opening argument."
        )
        response = await turn(
            "con_debater",
            f"Debate Topic: {topic}. Present your opening argument. Your opponent opened with: {response}",
        )

        # Remaining rounds
        for _ in range(num_rounds - 1):
            response = await turn(
                "pro_debater",
                f"Your opponent said: {response}. Present your next argument.",
            )
            response = await turn(
                "con_debater",
                f"Your opponent said: {response}. Present your next argument.",
            )

        return debate

    async def judge_debate(self, topic: str, debate_text: str) -> DebateEval:
        # prompt adapted from InspireScore: https://github.com/fywang12/InspireDebate/blob/main/inspirescore.py

        system_prompt = """
        You are an experienced debate judge tasked with evaluating debates. For each debate, you will assess both sides based on four key criteria: Emotional Appeal, Clarity of Argument and Reasoning, Logical Arrangement of Arguments, and Relevance to Debate Topic.

        For each of the four subdimensions, provide a score from 0 to 1 (with 0 being the lowest and 1 being the highest) for both the **Pro (Affirmative)** side and the **Con (Negative)** side. Additionally, provide a brief analysis for both sides for each subdimension.

        Scoring Criteria:
            1. **Emotional Appeal**
                - How effectively does each side connect with the audience emotionally? Does the argument evoke empathy, passion, or values?
                - **0**: No emotional appeal. The argument feels cold or disconnected.
                - **1**: Highly engaging emotionally, strongly connects with the audience.

            2. **Clarity of Argument and Reasoning**
                - Are the arguments clearly presented? Is the reasoning sound and easy to follow?
                - **0**: The arguments are unclear or confusing.
                - **1**: The arguments are well-structured and easy to understand.

            3. **Logical Arrangement of Arguments**
                - Is the argument presented in a logical, coherent manner? Does each point flow into the next without confusion?
                - **0**: The arguments are disorganized and difficult to follow.
                - **1**: The arguments follow a clear and logical progression.

            4. **Relevance to Debate Topic**
                - Does each argument directly address the debate topic? Are there any irrelevant points or off-topic distractions?
                - **0**: Arguments that stray far from the topic.
                - **1**: Every argument is focused and relevant to the topic.

        Please output the result in the following format:

        1. **Pro (Affirmative Side) Score**:
            - Emotional Appeal: [score]
            - Argument Clarity: [score]
            - Argument Arrangement: [score]
            - Relevance to Debate Topic: [score]
            - **Total Score**: [total score]

        2. **Con (Negative Side) Score**:
            - Emotional Appeal: [score]
            - Argument Clarity: [score]
            - Argument Arrangement: [score]
            - Relevance to Debate Topic: [score]
            - **Total Score**: [total score]

        3. **Winner**: [Pro/Con]
        4. **Reason**: [Provide detailed analysis based on the scores]
        """

        user_prompt = f"""
        Evaluate the debate on the topic: '{topic}'
        Debate analysis process and arguments are as follows:
        {debate_text}
        Provide a JSON formatted response with scores and comments for each criterion for both debaters.
        """

        response = self._client.models.generate_content(
            model="gemini-2.5-flash",
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=DebateEval,
            ),
            contents=user_prompt,
        )
        return response.parsed


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A debate judge.")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=9019, help="Port to bind the server"
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
        agent = DebateJudge()
        executor = GreenExecutor(agent)
        agent_card = debate_judge_agent_card("DebateJudge", agent_url)

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
