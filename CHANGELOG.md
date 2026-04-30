# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-04-30

### Added
- `ai.client`: `AIClient` abstract base + concrete implementations for Anthropic, OpenAI, Google Gemini, and Ollama
- `ai.client`: `AIResponse` dataclass with `estimated_cost_usd` property and `MODEL_COSTS` table
- `ai.client`: `ProviderError` — provider-agnostic exception with `status_code`, `retryable`, and `public_message`
- `ai.client`: `get_ai_client()` factory — selects provider by name, falls back to env vars for API keys
- `http.sse`: `SSEEvent` dataclass — formats Server-Sent Events with auto JSON serialization for dicts/lists
- `http.sse`: `sse_stream()` — wraps an async generator of `SSEEvent` into a FastAPI/Starlette `StreamingResponse`
- Optional dependency extras: `ai-anthropic`, `ai-openai`, `ai-google`, `ai-ollama`, `ai` (all providers), `http`
- `aws.lambda_logger`: `get_lambda_logger()` — structured JSON logger; level controlled via `LOG_LEVEL` env var

## [0.1.0] - 2026-04-24

### Added
- `ddd` module with DDD building blocks ported from ddd-python-aws
- `base_types`: `Entity`, `EntityId`, `RootEntity`, `DomainAggregate`, `RepositoryAggregate`,
  `ValueObject`, `DomainEvent`, `Command`, `EpochTime`, `Key`, `NamedEnum`, `Settings`, `Country`
- `lambda_logger`: structured Lambda-compatible logger
- `adapters.event_publisher`: `EventPublisher` protocol + `EventBridgePublisher` implementation
- `adapters.unit_of_work`: `UnitOfWork` protocol + `DynamoDbUnitOfWork` with SINGLE/BATCH modes
- `adapters.persistence.commons`: `Repository`, `SessionDB`, `WriteOperation` protocols
- `adapters.persistence.dynamodb_repository`: `DynamoDbRepository` for single-table pattern
- Project scaffolding: `aws`, `ai`, `parsers`, `http` module stubs
