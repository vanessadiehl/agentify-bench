## Agentify Bench: Domain-Adaptive Language Understanding Benchmark

Adapted from [AgentBeats](https://github.com/agentbeats/agentbeats)


## Project Structure
```
agentify-bench/
src/
└─ agentbeats/
   ├─ green_executor.py        # base A2A green agent executor
   ├─ models.py                # pydantic models for green agent IO
   ├─ client.py                # A2A messaging helpers
   ├─ client_cli.py            # CLI client to start assessment
   └─ run_scenario.py          # run agents and start assessment

scenarios/
└─ debate/                     # implementation of the debate example
   ├─ debate_judge.py          # green agent impl using the official A2A SDK
   ├─ adk_debate_judge.py      # alternative green agent impl using Google ADK
   ├─ debate_judge_common.py   # models and utils shared by above impls
   ├─ debater.py               # debater agent (Google ADK)
   └─ scenario.toml            # config for the debate example
```

Instructions:
1. Clone (or fork) the repo:
```
git clone https://github.com/your-team/agentify-bench.git`
cd agentify-bench
```
2. Install dependencies
```
uv sync
```
3. Set environment variables
```
cp sample.env .env
```
Add your Google API key to the .env file

