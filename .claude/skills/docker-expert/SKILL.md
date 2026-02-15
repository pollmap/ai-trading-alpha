---
name: docker-expert
description: "Advanced Docker containerization — Dockerfile optimization, multi-stage builds, Compose orchestration, security hardening, and production deployment. Use when working with Docker or Docker Compose."
---

# Docker Expert

Comprehensive containerization skill for Dockerfile optimization, security, and Docker Compose orchestration.

## Use this skill when

- Optimizing Dockerfiles (multi-stage builds, layer caching)
- Configuring Docker Compose services (depends_on, healthchecks, networks)
- Hardening container security (non-root users, secrets, capabilities)
- Setting up development vs production configurations
- Debugging container networking or volume issues
- Deploying multi-service stacks (API + worker + DB + proxy)

## Core Patterns

### Multi-Stage Build (Python)
```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
RUN groupadd -r app && useradd -r -g app app && chown -R app:app /app
USER app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
```

### Multi-Stage Build (Next.js)
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public
ENV NODE_ENV=production
EXPOSE 3000
CMD ["node", "server.js"]
```

### Compose Service Separation
```yaml
services:
  api:          # FastAPI — handles HTTP requests
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    depends_on:
      postgres: { condition: service_healthy }
      redis: { condition: service_healthy }

  worker:       # Background job processor
    command: python scripts/simulation_worker.py
    depends_on: [api]

  frontend:     # Next.js
    depends_on: [api]

  nginx:        # Reverse proxy + SSL
    depends_on: [api, frontend]
```

### Security Checklist
- [ ] Non-root USER with specific UID/GID
- [ ] No secrets in ENV — use Docker secrets or .env files
- [ ] Minimal base image (slim/alpine/distroless)
- [ ] .dockerignore excludes .git, node_modules, __pycache__, .env
- [ ] HEALTHCHECK defined for each service
- [ ] Resource limits (memory, CPU) set
- [ ] Read-only filesystem where possible

### Nginx as Reverse Proxy
```nginx
upstream api { server atlas-api:8000; }
upstream frontend { server atlas-frontend:3000; }

location /api/ { proxy_pass http://api; proxy_set_header Host $host; }
location / { proxy_pass http://frontend; }
```

## Key Principles

1. **Layer optimization**: Combine RUN commands, copy dependencies before source code
2. **Multi-stage**: Separate build dependencies from runtime
3. **Health checks**: Use `depends_on: { condition: service_healthy }`
4. **Non-root**: Always run as non-root user in production
5. **Compose profiles**: Use profiles for dev vs prod configurations

## References

- Source: [antigravity-awesome-skills/docker-expert](https://github.com/sickn33/antigravity-awesome-skills/tree/main/skills/docker-expert)
