# erns-shared

Shared Python utilities for AWS + AI projects. Provides reusable Domain-Driven Design (DDD) primitives, AWS service wrappers, and Lambda infrastructure patterns used across Lab products and Domenxa.

> Only code that has been reused in 2+ real projects enters this library.

## Installation

```bash
uv add erns-shared
# or
pip install erns-shared
```

## Modules

| Module                | Status         | Description                                                        |
| --------------------- | -------------- | ------------------------------------------------------------------ |
| `erns_shared.ddd`     | ✅ Available   | DDD primitives, DynamoDB UoW, EventBridge publisher                |
| `erns_shared.aws`     | ✅ Available   | S3, DynamoDB query helpers, SSM Parameter Store, Lambda Powertools |
| `erns_shared.ai`      | 🔲 Coming soon | Claude client with retry and cost logging                          |
| `erns_shared.parsers` | 🔲 Coming soon | PDF extraction, CSV parsing                                        |
| `erns_shared.http`    | 🔲 Coming soon | API Gateway response builders                                      |

---

## `erns_shared.ddd`

Battle-tested building blocks for DDD-style Python services on AWS.

### Define a domain model

```python
from erns_shared.ddd import (
    EntityId, DomainAggregate, DomainEvent, Command,
    ValueObject, EpochTime, update_last_update_date,
)

class OrderId(EntityId):
    value: str

class OrderPlaced(DomainEvent):
    domain_name: str = "orders"
    order_id: str
    total: float

class Order(DomainAggregate):
    id: OrderId
    total: float
    status: str = "pending"

    @update_last_update_date
    def place(self) -> None:
        self.status = "placed"
        self.add_event(OrderPlaced(order_id=self.id._key(), total=self.total))
```

### Persist with DynamoDB (single-table) + publish to EventBridge

```python
from erns_shared.ddd.adapters.unit_of_work import DynamoDbUnitOfWork
from erns_shared.ddd.adapters.persistence.dynamodb_repository import DynamoDbRepository

uow = DynamoDbUnitOfWork()
repo = DynamoDbRepository(session=uow.session, table_name="my-table", entity_type=Order)

order = Order(id=OrderId(value="ord-123"), total=99.99)
order.place()

with uow.transaction():
    repo.put(order)
    uow.publish_events(order.pull_events())
```

### Structured Lambda logger

```python
from erns_shared.ddd import get_lambda_logger

logger = get_lambda_logger()
logger.info("Order placed")
```

---

## `erns_shared.aws`

### S3Client

```python
from erns_shared.aws import S3Client

s3 = S3Client()

s3.upload("my-bucket", "docs/report.pdf", body=pdf_bytes, content_type="application/pdf")
data = s3.download("my-bucket", "docs/report.pdf")

if s3.key_exists("my-bucket", "docs/report.pdf"):
    url = s3.presigned_get_url("my-bucket", "docs/report.pdf", expiration=3600)

for key in s3.list_keys("my-bucket", prefix="docs/"):
    print(key)
```

### DynamoDBTable

High-level query helpers for the single-table pattern. Works with plain dicts — complements the DDD persistence layer.

```python
from boto3.dynamodb.conditions import Attr, Key
from erns_shared.aws import DynamoDBTable

table = DynamoDBTable("my-table")

# query — flexible sort key conditions
orders = table.query_by_pk("pk", "user#1", sk_condition=Key("sk").begins_with("order#"))
recent  = table.query_by_pk("pk", "user#1", sk_condition=Key("sk").between("order#2024", "order#2025"))

# query + filter on non-key attributes
shipped = table.query_by_pk(
    "pk", "user#1",
    sk_condition=Key("sk").begins_with("order#"),
    filter_expression=Attr("status").eq("shipped"),
)

# convenience prefix helper
orders = table.query_by_pk_sk_prefix("pk", "user#1", "sk", "order#")

# scan the full table
active = table.scan(filter_expression=Attr("active").eq(True))

# batch read (auto-chunks at 100)
items = table.batch_get(keys=[{"pk": "user#1", "sk": "profile"}])

# single-item writes with optional condition + return previous value
old = table.put_item(item, return_values="ALL_OLD")
table.put_item(item, condition=Attr("version").eq(3))
table.delete_item(key, condition=Attr("status").eq("pending"))
```

#### Batch writer — not atomic, auto-chunks at 25 per call

```python
with table.batch_writer() as w:
    w.put({"pk": "user#1", "sk": "profile", "name": "Alice"})
    w.delete({"pk": "user#old", "sk": "profile"})
# flushes on exit, clears ops on exception
```

#### Transaction writer — fully atomic, max 100 operations

```python
with table.transaction_writer() as w:
    w.put({"pk": "order#1", "sk": "meta", "status": "placed"})
    w.put({"pk": "order#1", "sk": "meta"}, condition=Attr("version").eq(2))
    w.delete({"pk": "draft#1", "sk": "meta"})
# all land or none do
```

### SSMClient

```python
from erns_shared.aws import SSMClient

ssm = SSMClient()

# single parameter — cached by default
db_url = ssm.get_parameter("/app/db_url")

# load an entire path into cache in one call
params = ssm.get_parameters_by_path("/app/prod")

# write
ssm.put_parameter("/app/feature_flag", "true", param_type="String", overwrite=True)

# cache management
ssm.invalidate_cache("/app/db_url")  # single key
ssm.invalidate_cache()               # full flush
```

### Lambda Powertools

```python
from erns_shared.aws import build_powertools, LambdaContext

logger, tracer, metrics = build_powertools(service="order-service", namespace="MyApp")
# falls back to POWERTOOLS_SERVICE_NAME / POWERTOOLS_METRICS_NAMESPACE env vars

@tracer.capture_lambda_handler
@logger.inject_lambda_context
def handler(event: dict, context: LambdaContext) -> dict:
    logger.info("Handling event")
    return {"statusCode": 200}
```

---

## Requirements

- Python 3.14+
- `pydantic >= 2.0`
- `pydantic-settings >= 2.0`
- `boto3 >= 1.34`
- `backoff >= 2.0`
- `aws-lambda-powertools >= 2.0`

## Contributing

See [GUIDE.md](GUIDE.md) for setup instructions, project structure, conventions, and the release workflow.
