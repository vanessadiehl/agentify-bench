from pydantic import BaseModel
from typing import Optional

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)


class EntityMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


class RelationshipMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


class TurnScore(BaseModel):
    turn: int
    entity_f1: Optional[float] = None
    relationship_f1: Optional[float] = None
    consistency: Optional[float] = None
    entity_metrics: Optional[dict] = None
    rel_metrics: Optional[dict] = None
    parsed_ok: bool = True
    error: Optional[str] = None


class EpisodeResult(BaseModel):
    episode_id: str
    episode_title: str
    avg_entity_f1: float
    avg_rel_f1: float
    avg_persistence: float
    turns: list[TurnScore]


class GlobalMetrics(BaseModel):
    avg_entity_f1: float
    avg_rel_f1: float
    avg_persistence: float
    num_episodes: int
    total_turns: int


class SemanticEvalResult(BaseModel):
    global_metrics: GlobalMetrics
    episodes: list[EpisodeResult]


def semantic_judge_agent_card(agent_name: str, card_url: str) -> AgentCard:
    skill = AgentSkill(
        id='evaluate_crm_mapping',
        name='Evaluates CRM ontology mapping',
        description='Evaluate how well agents map legal cases to CRM entities and relationships across multiple turns and episodes.',
        tags=['crm', 'ontology', 'legal', 'multi-turn', 'multi-episode'],
        examples=["""
{
  "participants": {
    "crm_mapper": "https://crm-mapper.example.com:443"
  },
  "config": {
    "episodes": [
      "scenarios/domain_adapt_crm/episodes/legal_to_crm.yml",
      "scenarios/domain_adapt_crm/episodes/employment_discrimination.yml",
      "scenarios/domain_adapt_crm/episodes/contract_breach.yml"
    ]
  }
}
"""]
    )
    agent_card = AgentCard(
        name=agent_name,
        description='Evaluate CRM ontology mapping across multi-turn, multi-episode legal case analysis with entity, relationship, and consistency scoring.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card