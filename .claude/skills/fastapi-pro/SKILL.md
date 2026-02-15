---
name: fastapi-pro
description: "Build high-performance async APIs with FastAPI, SQLAlchemy 2.0, Pydantic V2, and OAuth2/JWT authentication. Use when building FastAPI endpoints, implementing OAuth flows, or designing API architecture."
---

# FastAPI-Pro

Comprehensive FastAPI expert skill for high-performance, async-first API development with modern Python patterns.

## Use this skill when

- Building FastAPI applications or adding new API endpoints
- Implementing OAuth2/JWT authentication flows
- Designing REST API contracts with Pydantic models
- Setting up SQLAlchemy 2.0 async database access
- Optimizing API performance (caching, pooling, rate limiting)
- Writing async tests with pytest-asyncio

## Do not use this skill when

- Working on frontend-only tasks
- Working with Django or Flask (different frameworks)

## Core Patterns

### Project Structure
```
src/api/
├── main.py           # FastAPI app, lifespan, middleware
├── deps.py           # Dependency injection (DB session, current user)
├── middleware.py      # JWT auth, rate limiting, tenant context
├── auth/
│   ├── oauth.py      # OAuth2 flows (Google, GitHub)
│   └── providers.py  # Provider configurations
├── routes/           # Endpoint modules
├── models/           # Pydantic request/response schemas
└── db/               # Repository pattern (SQLAlchemy async)
```

### Authentication (OAuth2 + JWT)
- Use `authlib` for OAuth2 client flows
- HMAC-signed state parameter for CSRF protection (Redis TTL 10min)
- JWT in httpOnly cookie (web) + Authorization header (API clients)
- Middleware extracts tenant from JWT, injects `request.state.tenant_id`

### Async Database Access
```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

engine = create_async_engine(db_url, pool_size=10, max_overflow=20)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(engine) as session:
        yield session
```

### Pydantic V2 Models
- Use `model_config = ConfigDict(from_attributes=True)` for ORM mode
- Separate Create/Update/Response schemas
- Use `Field(...)` for validation constraints

### Error Handling
```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
```

### Testing
```python
from fastapi.testclient import TestClient
# or async:
from httpx import AsyncClient, ASGITransport

async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
    response = await ac.get("/api/health")
```

## Key Principles

1. **Async-first**: All IO operations use async/await
2. **Pydantic contracts**: Design API schema before implementation
3. **Dependency injection**: Use `Depends()` for DB sessions, auth, rate limits
4. **Repository pattern**: Separate data access from route handlers
5. **Structured logging**: Use structlog, not print() or stdlib logging
6. **Type safety**: 100% type hints, no `Any`
7. **Security**: Parameterized queries, input validation, CORS restrictions

## References

- Source: [antigravity-awesome-skills/fastapi-pro](https://github.com/sickn33/antigravity-awesome-skills/tree/main/skills/fastapi-pro)
