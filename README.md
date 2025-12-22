## Agentify-Bench: Multi-Turn Semantic Evaluation for Domain-Adaptive CRM Mapping
This benchmark evaluates the AI agents' ability to map legal case descriptions to CRM ontology structures across multiple conversation turns.

## Overview
AgentifyBench addresses a critical gap in AI agent evaluation: can agents correctly extract relationships in different domains?
Existing benchmarks evaluate agents on isolated tasks; it is our understanding that real-world agents require:
- Domain adaptation: understanding context is essential to agents extracting and mapping information in multidomain understanding.
- Multi-turn consistency: Maintaining semantic understanding when context changes
- Structural reasoning: Extracting entities and relationships, not relying on surface patterns.

AgentifyBench tests all three by evaluating agents' ability to extract CRM entities (Account, Contact, Case, Property, Event, Interaction) and link to relationships from legal cases descriptions across 3 conversion turns in order to test true semantic eval. 

## The Problem
While LLM-based agents are increasingly capable of extracting raw information, their precision remains a challenge. 
- Legal and CRM teams require agents that can autonomously and accurately populate CRM fields directly from legal documentation.
- Agents often struggle to maintain consistency when presented with corrections, updates, and new information.
- Current benchmarks fail to measure agents' "staying power" in a conversation or its ability to adapt as the CRM/legal case progresses


## Solution: Multi-turn Evaluation Framework
Agentify-Bench utilizes a structured evaluation protocol consisting of three distinct episodes that mimic high-complexity legal-CRM workflows, each spanning three conversational turns. This approach allows for objective F1 weighting scoring of an agent's semantic reasoning. 
Episodes:
1. Construction Defect Case - Tests basic entity/relationship extraction and multi-turn persistence
2. Employment Discrimination Case - Tests multi-party relationships and role updates
3. Commercial Contract Breach - Tests financial property tracking and personnel transitions

## Scoring Dimensions:
- Entity F1: Can the agent extract correct entity types and names?
- Relationship F1: Can the agent map entities to correct relationship types?
- Persistence: Does the agent maintain previous relationships when the context changes?

# Quick Start
## Prerequisites
- Python 3.11+
- Docker (optional, for containerized runs)
- Google API key for Gemini API


## Local Setup

```
bash
git clone https://github.com/your-team/agentify-bench.git
cd agentify-bench

uv sync
cp sample.env .env      # add your GOOGLE_API_KEY, etc.

```
## Run Benchmark Locally

```
# Run benchmark on terminal:

uv run agentbeats-run scenarios/domain_adapt_crm/scenario.toml

```
Optional: Start all three separately

```
# Terminal 1: Start green agent (judge)
uv run python scenarios/domain_adapt_crm/semantic_judge.py --host 127.0.0.1 --port 9009

# Terminal 2: Start purple agent (baseline mapper)
uv run python scenarios/domain_adapt_crm/semantic_white_baseline.py --host 127.0.0.1 --port 9019

# Terminal 3: Run benchmark
uv run agentbeats-run scenarios/domain_adapt_crm/scenario.toml

```

## Run Benchmark With Docker

```
# Build image
docker build -t agentify-bench:latest .

# Run benchmark
docker run -e GOOGLE_API_KEY="your-api-key" agentify-bench:latest

```

---


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

scenarios/                     # reference implementation, debate style evaluator
├── debate/
│   ├── adk_debate_judge.py
│   ├── debate_judge_common.py
│   ├── debate_judge.py
│   ├── debater.py
│   └── scenario.toml
│
└── domain_adapt_crm/              # benchmark    
    ├── episodes/                  # test cases
    ├── scenario.toml              # orchestrate green and purple agents  
    ├── semantic_judge_common.py   # shared models
    ├── semantic_judge.py          # baseline CRM mapper
    └── semantic_white_baseline.py # green agent - evaluates CRM mapper
```
## Acknowledgments
Built on Google's A2A protocol and AgentBeats framework. Special thanks to the Berkeley RDI team for the competition structure.
