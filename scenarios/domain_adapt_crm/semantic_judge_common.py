from pydantic import BaseModel
from typing import Literal

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
    entity_f1: float
    relationship_f1: float
    consistency: float | None


class SemanticEvalResult(BaseModel):
    per_turn_scores: list[TurnScore]
    avg_entity_f1: float
    avg_relationship_f1: float
    avg_consistency: float | None
    learning_trajectory: str


def semantic_judge_agent_card(agent_name: str, card_url: str) -> AgentCard:
    skill = AgentSkill(
        id='evaluate_crm_mapping',
        name='Evaluates CRM ontology mapping',
        description='Evaluate how well agents map legal cases to CRM entities and relationships across multiple turns.',
        tags=['crm', 'ontology', 'legal', 'multi-turn'],
        examples=["""
{
  "participants": {
    "crm_mapper": "https://crm-mapper.example.com:443"
  },
  "config": {
    "episodes": ["scenarios/domain_adapt_crm/episodes/legal_to_crm.yml"]
  }
}
"""]
    )
    agent_card = AgentCard(
        name=agent_name,
        description='Evaluate CRM ontology mapping across multi-turn legal case analysis with entity, relationship, and consistency scoring.',
        url=card_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )
    return agent_card