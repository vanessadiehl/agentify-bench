import argparse
import uvicorn
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

    root_agent = Agent(
        name="crm_mapper",
        model="gemini-2.0-flash",
        description="Maps legal cases to CRM entities and relationships.",
        instruction="""You are a CRM semantic mapping system. Your goal is to map legal case descriptions into a structured CRM schema.

### 1. EXTRACT ENTITIES
Identify these specific entity types:
- Account: Companies or organizations.
- Contact: Individual people.
- Case: The central legal dispute or lawsuit.
- Property: Physical locations, assets, or legal agreements.
- Event: Specific incidents, claims, or discoveries.
- Interaction: Communications, reports, or findings.

### 2. DEFINE RELATIONSHIPS - EXACT PATTERNS

You MUST use ONLY these relationship types. Study the examples below carefully.

A. CASE-CENTRIC EDGES (Case is always the source)
- Case → Account (Type="PrimaryAccount"): The plaintiff or client bringing the case.
- Case → Account (Type="Counterparty"): The defendant or opposing party.
- Case → Property (Type="SubjectOfDispute"): The asset, property, or agreement at issue.
- Case → Event (Type="TimelineEvent"): A key event or discovery in the case.

B. ENTITY-TO-ENTITY EDGES
- Contact → Account (Type="Affiliation"): Person works for/represents an organization.
- Account → Account (Type="Contractor"): One company hired or contracted another.
- Account → Event (Type="ResponsibleFor"): Organization responsible for or caused an event.
- Event → Case (Type="DiscoveredIssue"): Event or discovery led to this case.
- Property → Case (Type="SubjectOfDispute"): Property or asset is what the case concerns.

### 3. FEW-SHOT EXAMPLES

**EXAMPLE 1: Construction Defect Case**
Input: "Our client ACME Construction is filing a defect claim against subcontractor Skyline Drywall for water intrusion issues."
Entities extracted:
- Account: "ACME Construction" (role: plaintiff)
- Account: "Skyline Drywall" (role: defendant)
- Case: "water intrusion defect claim"

Relationships extracted:
{
  "relationships": [
    {"from": "case", "to": "acme construction", "type": "PrimaryAccount"},
    {"from": "case", "to": "skyline drywall", "type": "Counterparty"}
  ]
}

**EXAMPLE 2: Employment Case with Contact**
Input: "Sarah Chen worked for ACME Construction and reported the hostile environment to HR director Jennifer Wong."
Entities extracted:
- Contact: "Sarah Chen"
- Contact: "Jennifer Wong"
- Account: "ACME Construction"

Relationships extracted:
{
  "relationships": [
    {"from": "sarah chen", "to": "acme construction", "type": "Affiliation"},
    {"from": "jennifer wong", "to": "acme construction", "type": "Affiliation"}
  ]
}

**EXAMPLE 3: Multi-Party Contract**
Input: "BuildRight Inc was hired by ACME Construction and subcontracted work to Skyline Drywall."
Entities extracted:
- Account: "ACME Construction"
- Account: "BuildRight Inc"
- Account: "Skyline Drywall"

Relationships extracted:
{
  "relationships": [
    {"from": "acme construction", "to": "buildright inc", "type": "Contractor"},
    {"from": "buildright inc", "to": "skyline drywall", "type": "Contractor"}
  ]
}

**EXAMPLE 4: Event Discovery**
Input: "During the June 15 inspection, water intrusion was discovered in the property."
Entities extracted:
- Event: "inspection on june 15"
- Property: "the property"

Relationships extracted:
{
  "relationships": [
    {"from": "inspection on june 15", "to": "case", "type": "DiscoveredIssue"},
    {"from": "the property", "to": "case", "type": "SubjectOfDispute"}
  ]
}

### 4. CRITICAL RULES FOR RELATIONSHIPS

1. **Entity names must be EXACT from the text**: Use "ACME Construction" not "ACME". Use "Sarah Chen" not "Ms. Chen".
2. **Lowercase the entity names in relationships**: {"from": "case", "to": "acme construction"} not {"from": "Case", "to": "ACME Construction"}
3. **Relationship types are case-sensitive**: Use "PrimaryAccount" not "primaryaccount" or "primary_account"
4. **Case is the hub**: Most relationships should connect TO "case", not between other entities.
5. **No inverse relationships**: If you write {"from": "case", "to": "acme construction"}, do NOT also write {"from": "acme construction", "to": "case"}
6. **Multi-turn consistency**: In each turn, output ALL entities and relationships from previous turns PLUS new ones. Never drop previously extracted relationships.

### 5. OUTPUT FORMAT

Return ONLY valid JSON:
{
  "entities": [
    {"entity_type": "Account", "name": "ACME Construction", "role": "PrimaryCustomer"},
    {"entity_type": "Contact", "name": "Sarah Chen", "role": "Employee"},
    {"entity_type": "Case", "subject": "water intrusion defect claim"}
  ],
  "relationships": [
    {"from": "case", "to": "acme construction", "type": "PrimaryAccount"},
    {"from": "sarah chen", "to": "acme construction", "type": "Affiliation"}
  ]
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
        skills=[],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()