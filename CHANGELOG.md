# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `aws.s3`: `S3Client` — upload, download, delete, `key_exists`, `list_keys`, presigned GET/PUT URLs
- `aws.dynamodb`: `DynamoDBTable` — single-table query helpers (`query_by_pk`, `query_by_pk_sk_prefix`, `batch_get`, `put_item`, `delete_item`)
- `aws.ssm`: `SSMClient` — SSM Parameter Store with in-process cache (`get_parameter`, `get_parameters_by_path`, `put_parameter`, `invalidate_cache`)
- `aws.powertools`: re-exports `Logger`, `Tracer`, `Metrics`, `MetricUnit`, `LambdaContext` from `aws-lambda-powertools`; `build_powertools()` factory for consistent Lambda setup
- Added `aws-lambda-powertools>=2.0` runtime dependency
- Added `moto[s3,dynamodb,ssm]>=5.0` dev dependency

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
