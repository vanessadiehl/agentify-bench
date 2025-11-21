import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Dict

import tomllib

from agentbeats.client import send_message
from agentbeats.models import EvalRequest
from a2a.types import (
    AgentCard,
    Message,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    TaskState,
    Part,
    TextPart,
    DataPart,
)


def parse_toml(d: dict[str, object]) -> tuple[EvalRequest, str]:
    green = d.get("green_agent")
    if not isinstance(green, dict) or "endpoint" not in green:
        raise ValueError("green.endpoint is required in TOML")

    green_endpoint: str = green["endpoint"]

    # collect participants
    parts: dict[str, str] = {}
    for p in d.get("participants", []):
        if isinstance(p, dict):
            role = p.get("role")
            endpoint = p.get("endpoint")
            if role and endpoint:
                parts[role] = endpoint

    eval_req = EvalRequest(
        participants=parts,
        config=d.get("config", {}) or {}
    )
    return eval_req, green_endpoint

def print_parts(parts, task_state: str | None = None):
    text_parts = []
    data_parts = []

    for part in parts:
        if isinstance(part.root, TextPart):
            try:
                data_item = json.loads(part.root.text)
                data_parts.append(data_item)
            except Exception:
                text_parts.append(part.root.text.strip())
        elif isinstance(part.root, DataPart):
            data_parts.append(part.root.data)

    output = []
    if task_state:
        output.append(f"[Status: {task_state}]")
    if text_parts:
        output.append("\n".join(text_parts))
    if data_parts:
        output.extend(json.dumps(item, indent=2) for item in data_parts)

    print("\n".join(output) + "\n")

# ADDED: Track if we've received artifacts
artifacts_received = False

async def event_consumer(event, card: AgentCard):
    global artifacts_received
    match event:
        case Message() as msg:
            print_parts(msg.parts)

        case (task, TaskStatusUpdateEvent() as status_event):
            status = status_event.status
            parts = status.message.parts if status.message else []
            print_parts(parts, status.state.value)
            if status.state.value == "completed":
                if task.artifacts:
                    print("[Status: artifacts]")
                    for artifact in task.artifacts:
                        print_parts(artifact.parts)
                        artifacts_received = True

        case (task, TaskArtifactUpdateEvent() as artifact_event):
            print("[Status: Artifact update]")
            print_parts(artifact_event.artifact.parts)
            artifacts_received = True

        case task, None:
            status = task.status
            parts = status.message.parts if status.message else []
            print_parts(parts, task.status.state.value)
            if task.artifacts:
                print("[Status: artifacts]")
                for artifact in task.artifacts:
                    print_parts(artifact.parts)
                    artifacts_received = True

        case _:
            print("Unhandled event")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client_cli.py <scenario.toml>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    toml_data = path.read_text()
    data = tomllib.loads(toml_data)

    req, green_url = parse_toml(data)

    msg = req.model_dump_json()
    await send_message(msg, green_url, streaming=True, consumer=event_consumer)
    
    # ADDED: Wait a bit for any trailing artifacts
    await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())