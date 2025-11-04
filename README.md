## Agentify Bench: Domain-Adaptive Language Understanding Benchmark

Adapted from [AgentBeats](https://github.com/agentbeats/agentbeats)

## Overview

This repo is a playground for **AgentBeats-style green agents** built by **Team Agentify Bennch**.

Goal: Build and compare **multi-agent benchmarks** that focus on
- **Domain-adaptive language understanding** (mapping concepts across CRM, Legal, Support, etc.)
- **Standard A2A interfaces** so any compatible agent can plug in
- Reproducible, leaderboard-style evaluation

We reuse the core AgentBeats structure (`src/agentbeats/*`) and add our own scenarios under `scenarios/`.

---

## Scenarios

Right now the repo contains:

- `scenarios/debate/`  
  - Original debate example from AgentBeats (sanity check / reference).
- `scenarios/domain_adapt_crm/` *(Need to build)*  
  - Our **Domain-Adaptive Language Understanding Benchmark**  
  - Green agent: semantic “judge” that evaluates how well a white agent maps Legal → CRM ontology  
  - White agent: baseline participant (can be swapped out for any A2A-compatible agent)

As we build more, each scenario will live in its own folder with:
- a green agent implementation,
- one or more white agents,
- and a `scenario.toml` config file.

---

## Running Scenarios

### 1. Debate (baseline sanity check)

```
bash
git clone https://github.com/your-team/agentify-bench.git
cd agentify-bench

uv sync
cp sample.env .env      # add your GOOGLE_API_KEY, etc.

uv run python src/agentbeats/run_scenario.py \
  --scenario scenarios/debate/scenario.toml
```

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

