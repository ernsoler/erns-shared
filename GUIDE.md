# erns-shared — Contributor Guide

> Shared Python utilities for AWS + AI projects. Used across Lab products and Domenxa.
> **Rule**: only code that has been reused in 2+ real projects enters this library.

---

## Stack

- **Python** 3.11+
- **uv** — dependency management + publishing
- **PyPI** — public registry
- **GitHub Actions** — CI on every PR, publish on GitHub Release

---

## Setup

```bash
# Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
uv sync
```

---

## Project Structure

```
erns-shared/
├── src/
│   └── erns_shared/
│       ├── __init__.py
│       ├── py.typed                        # PEP 561 — typed package
│       ├── ddd/                            # DDD primitives
│       │   ├── __init__.py
│       │   ├── base_types.py               # Entity, EntityId, DomainAggregate, etc.
│       │   └── adapters/
│       │       ├── event_publisher.py      # EventBridgePublisher
│       │       ├── unit_of_work.py         # DynamoDbUnitOfWork
│       │       └── persistence/
│       │           ├── commons.py          # Repository, SessionDB protocols
│       │           └── dynamodb_repository.py
│       ├── aws/
│       │   ├── __init__.py
│       │   ├── s3.py                       # S3Client
│       │   ├── dynamodb.py                 # DynamoDBTable
│       │   ├── ssm.py                      # SSMClient
│       │   ├── powertools.py               # build_powertools()
│       │   └── lambda_logger.py            # get_lambda_logger()
│       ├── ai/                             # stub — coming soon
│       ├── http/                           # stub — coming soon
│       └── parsers/                        # stub — coming soon
├── tests/
│   ├── test_ddd/
│   └── test_aws/
├── .github/
│   └── workflows/
│       ├── ci.yml                          # lint + test on every push/PR
│       └── publish.yml                     # publish to PyPI on GitHub Release
├── pyproject.toml
├── README.md
├── CHANGELOG.md
└── GUIDE.md
```

---

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=erns_shared --cov-report=term-missing

# Lint
uv run ruff check .

# Format check
uv run ruff format --check .
```

---

## Release Workflow

1. Bump version in `pyproject.toml`
2. Open a PR, get it merged to `main`
3. Go to **GitHub → Releases → Draft a new release**
4. Create a new tag matching the version (e.g. `v0.2.0`) and publish the release
5. The `publish.yml` workflow fires automatically and pushes to PyPI

---

## PyPI Setup (one-time)

1. Create account at https://pypi.org
2. **Account Settings → API Tokens → Add API Token** (scope: entire account for first publish)
3. Add to GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**
   - Name: `PYPI_API_TOKEN`
   - Value: `pypi-...`

---

## Conventions

- **Semver**: `MAJOR.MINOR.PATCH`
  - PATCH — bug fix, no API change
  - MINOR — new feature, backwards compatible
  - MAJOR — breaking change
- **Conventional Commits**: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`
- No module enters without tests
- Public API only — keep `__init__.py` clean, don't export internal helpers
- Type hints everywhere — this lib is imported by other projects

---

## Using Locally Before Publishing

```bash
# In the consuming project's directory
uv add --editable ../erns-shared
```

---

## Module Roadmap

| Module              | Status         | Description                                              |
| ------------------- | -------------- | -------------------------------------------------------- |
| `erns_shared.ddd`   | Done           | DDD primitives, DynamoDB UoW, EventBridge publisher      |
| `erns_shared.aws`   | Done           | S3, DynamoDB helpers, SSM, Lambda Powertools, logger     |
| `erns_shared.ai`    | Coming soon    | Claude client with retry and cost logging                |
| `erns_shared.parsers` | Coming soon  | PDF extraction, CSV parsing                              |
| `erns_shared.http`  | Coming soon    | API Gateway response builders                            |
