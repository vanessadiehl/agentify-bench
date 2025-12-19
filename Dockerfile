
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install uv

RUN uv sync

EXPOSE 9009 9019

CMD ["uv", "run", "agentbeats-run", "scenarios/domain_adapt_crm/scenario.toml"]
