## Agentify-Bench: Multi-Turn Semantic Evaluation for Domain-Adaptive CRM Mapping
This benchmark evaluates the AI agents' ability to map legal case descriptions to CRM ontology structures across multiple conversation turns.

## Overview
AgentifyBench addresses a critical gap in AI agent evaluation: can agents correctly extract relationships in different domains?
Existing benchmarks evaluate agents on isolated tasks; it is our understanding that real-world agents require:
- Domain adaptation: understanding context is essential to agents extracting and mapping information in multidomain understanding.
- Multi-turn consistency: Maintaining semantic understanding when context changes
- Structural reasoning: Extracting entities and relationships, not relying on surface patterns.

AgentifyBench tests all three by evaluating agents' ability to extract CRM entities (Account, Contact, Case, Property, Event, Interaction) and link to relationships from legal case descriptions across 3 conversation turns to test true semantic eval. 

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
- Docker 
- Google API key for Gemini API


## Run Benchmark With Docker
```bash
git clone https://github.com/your-team/agentify-bench.git
```

### Terminal 1: Build and Run Green Agent (Evaluator)
```bash
cd agentify-bench
export GOOGLE_API_KEY="your-actual-key"
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.green -t vanessa939/agentify-bench-green:latest --push .
docker run -e GOOGLE_API_KEY=$GOOGLE_API_KEY vanessa939/agentify-bench-green:latest
```

Wait for "Application startup complete" message.

### Terminal 2 (another tab): Build and Run Purple Agent (Baseline Mapper)
```bash
cd agentify-bench
export GOOGLE_API_KEY="your-actual-key"
docker buildx build --platform linux/amd64,linux/arm64 -f Dockerfile.purple -t vanessa939/agentify-bench-purple:latest --push .
docker run -e GOOGLE_API_KEY=$GOOGLE_API_KEY vanessa939/agentify-bench-purple:latest
```

Wait for "Application startup complete" message.

### Terminal 3(another tab): Run Benchmark
```bash
cd agentify-bench
export GOOGLE_API_KEY="your-actual-key"
uv run agentbeats-run scenarios/domain_adapt_crm/scenario.toml
```

This runs all 3 episodes with both agents and generates results.


## Local Setup
```bash
git clone https://github.com/your-username/agentify-bench.git
cd agentify-bench
uv sync
cp sample.env .env
# Edit .env and add your GOOGLE_API_KEY
```

## Run Benchmark Locally

### Quick Start (Recommended)
```bash
export GOOGLE_API_KEY="your-actual-key"
uv run agentbeats-run scenarios/domain_adapt_crm/scenario.toml
```

This runs all 3 episodes automatically.

### Manual Setup (Optional)

If you want to run agents separately:

**Terminal 1: Green Agent (Evaluator)**
```bash
cd agentify-bench
export GOOGLE_API_KEY="your-actual-key"
uv run python scenarios/domain_adapt_crm/semantic_judge.py --host 127.0.0.1 --port 9009
```

**Terminal 2: Purple Agent (Baseline Mapper)**
```bash
cd agentify-bench
export GOOGLE_API_KEY="your-actual-key"
uv run python scenarios/domain_adapt_crm/semantic_white_baseline.py --host 127.0.0.1 --port 9019
```

**Terminal 3: Run Benchmark**
```bash
cd agentify-bench
export GOOGLE_API_KEY="your-actual-key"
uv run agentbeats-run scenarios/domain_adapt_crm/scenario.toml
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

scenarios/
├── debate/
│   ├── adk_debate_judge.py
│   ├── debate_judge_common.py
│   ├── debate_judge.py
│   ├── debater.py
│   └── scenario.toml
│
└── domain_adapt_crm/
    ├── episodes/                  # test cases (3 legal domains)
    ├── scenario.toml              # orchestrate green and purple agents  
    ├── semantic_judge_common.py   # shared models
    ├── semantic_judge.py          # green agent - evaluates CRM mapping
    └── semantic_white_baseline.py # purple agent - baseline CRM mapper
```
## Acknowledgments
Built on Google's A2A protocol and AgentBeats framework. Special thanks to the Berkeley RDI team for the competition structure.
