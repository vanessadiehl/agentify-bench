import argparse
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from a2a.types import AgentCapabilities, AgentCard

def main():
    parser = argparse.ArgumentParser(description="Run the A2A CRM mapper agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    args = parser.parse_args()

    # Retrieve API key explicitly for stability
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    root_agent = Agent(
        name="crm_mapper",
        model="gemini-2.0-flash",
        description="Maps legal cases to CRM entities and relationships.",
        api_key=api_key,
        instruction="""You are a CRM semantic mapping system. Your goal is to map legal case descriptions into a structured CRM schema.

### 1. EXTRACT ENTITIES
Identify these specific entity types:
- Account: Companies or organizations.
- Contact: Individual people.
- Case: The central legal dispute.
- Property: Physical locations/assets.
- Interaction: Meetings, inspections, or discoveries.

### 2. DEFINE RELATIONSHIPS
You must strictly follow this schema. Use these EXACT types:

A. CASE-CENTRIC (Case is the source)
- Case → Account (Type="PrimaryAccount")
- Case → Account (Type="Counterparty")
- Case → Property (Type="SubjectProperty")
- Case → Interaction (Type="TimelineEvent")

B. ENTITY-TO-ENTITY
- Contact → Account (Type="EmployedBy")
- Interaction → Case (Type="DiscoveredIssue")
- Account → Interaction (Type="ResponsibleFor")
- Account → Account (Type="Hired")
- Account → Account (Type="ContractedBy")
- Property → Case (Type="SubjectProperty")

### 3. STRICT NAMING & PERSISTENCE RULES
- **STRICT NAMING:** Use the most complete version of the name found in the text (e.g., "ACME Construction" not "ACME"). These names are the keys for relationships.
- **CUMULATIVE OUTPUT:** In multi-turn conversations, your JSON must include ALL entities and relationships established in previous turns, plus any new ones.
- **IDENTITY STABILITY:** Once an entity is named, do not change its name in subsequent turns unless corrected. If "Sarah Chen" is identified, use "Sarah Chen" as the key in all relationships.

### 4. OUTPUT FORMAT
Return valid JSON only.
{
  "entities": [{"entity_type": "Account", "name": "ACME Construction", "role": "PrimaryCustomer"}],
  "relationships": [{"from": "Sarah Chen", "to": "ACME Construction", "type": "EmployedBy"}]
}
""",
    )

    agent_card = AgentCard(
        name="crm_mapper",
        description='Maps legal cases to CRM entities and relationships.',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()