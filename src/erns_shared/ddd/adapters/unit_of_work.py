import enum
import contextlib
import boto3
import backoff
from botocore import exceptions as boto3_exceptions
from typing import Protocol, List, Iterator, Dict, Final
import erns_shared.ddd.lambda_logger as lambda_logger
import erns_shared.ddd.base_types as base_types
from erns_shared.ddd.adapters import event_publisher
from erns_shared.ddd.adapters.persistence import commons as persistence_commons
from erns_shared.ddd.adapters.persistence.commons import WriteOperation

_LOGGER = lambda_logger.get_lambda_logger()


class UnknownTransactionTypeError(Exception):
    ...


class DuplicateWriteOperationsError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "A single write operation for each entity is allowed in the same context"
        )


class DynamoBatchSizePerTrxExceedsError(Exception):
    def __init__(self) -> None:
        super().__init__("DynamoDB only allows 100 operations in a single transaction")


class TransactionFailedError(Exception):
    ...


class TransactionType(base_types.NamedEnum):
    SINGLE = enum.auto()
    BATCH = enum.auto()
    NONE = enum.auto()


MAX_DYNAMO_DB_BATCH_SIZE_PER_TRX: Final = 100


class UnitOfWork(Protocol):
    @property
    def session(self) -> persistence_commons.SessionDB:
        ...

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        ...

    @contextlib.contextmanager
    def batch(self) -> Iterator[None]:
        ...

    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...

    def publish_events(self, events: List[base_types.DomainEvent]) -> None:
        ...


class DefaultDynamoDBSession:
    def __init__(self) -> None:
        self._batches: Dict[str, WriteOperation] = {}
        self.client = boto3.client("dynamodb")

    def add_write_operation(self, operation: WriteOperation) -> None:
        if self._batches.get(operation.id):
            raise DuplicateWriteOperationsError()
        self._batches[operation.id] = operation

    def clear_batches(self) -> None:
        self._batches.clear()

    @backoff.on_exception(backoff.fibo, TransactionFailedError, max_tries=3, max_time=4)
    def _persist_operations(self, operations: List[WriteOperation]) -> None:
        try:
            items = [op.entity_serialized for op in operations]
            self.client.transact_write_items(TransactItems=items)
        except boto3_exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            _LOGGER.exception(f"Transaction error {error_code}. Exception {str(e)}")
            if error_code == "TransactionCanceledException":
                raise TransactionFailedError(
                    ".".join(
                        reason["Message"] for reason in e.response["CancellationReasons"]
                    )
                )
            raise TransactionFailedError(error_code) from e
        except Exception as e:
            _LOGGER.exception(f"Transaction error. Exception {str(e)}")
            raise TransactionFailedError() from e

    def execute_in_single_transaction(self) -> None:
        if not self._batches:
            _LOGGER.info("[UoW]: No write operations to process")
            return
        if len(self._batches) > MAX_DYNAMO_DB_BATCH_SIZE_PER_TRX:
            raise DynamoBatchSizePerTrxExceedsError()
        self._persist_operations(operations=[*self._batches.values()])
        self.clear_batches()

    def execute_in_batch_transaction(self) -> None:
        if not self._batches:
            _LOGGER.info("[UoW]: No write operations to process")
            return
        chunks: Iterator[List[WriteOperation]] = base_types.split_list(
            input_list=[*self._batches.values()],
            chunk_size=MAX_DYNAMO_DB_BATCH_SIZE_PER_TRX,
        )
        for chunk in chunks:
            self._persist_operations(operations=chunk)
        self.clear_batches()


class DynamoDbUnitOfWork:
    def __init__(self) -> None:
        self._message_bus_client = event_publisher.EventBridgePublisher()
        self._session = DefaultDynamoDBSession()
        self._events_to_publish: List[base_types.DomainEvent] = []
        self._transaction_type: TransactionType = TransactionType.NONE

    @property
    def session(self) -> persistence_commons.SessionDB:
        return self._session  # type: ignore

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            _LOGGER.info("Init single transaction context manager")
            self._transaction_type = TransactionType.SINGLE
            yield
            self.commit()
        finally:
            self._transaction_type = TransactionType.NONE
            self._events_to_publish.clear()

    @contextlib.contextmanager
    def batch(self) -> Iterator[None]:
        try:
            _LOGGER.info("Init batch transaction context manager")
            self._transaction_type = TransactionType.BATCH
            yield
            self.commit()
        finally:
            self._transaction_type = TransactionType.NONE
            self._events_to_publish.clear()

    def commit(self) -> None:
        if self._transaction_type == TransactionType.SINGLE:
            self._session.execute_in_single_transaction()
        elif self._transaction_type == TransactionType.BATCH:
            self._session.execute_in_batch_transaction()
        else:
            raise UnknownTransactionTypeError(
                "Transaction type must be SINGLE or BATCH before committing"
            )
        if self._events_to_publish:
            _LOGGER.info("Publishing domain events")
            self._message_bus_client.publish(events=self._events_to_publish)

    def rollback(self) -> None:
        self._session.clear_batches()
        self._events_to_publish.clear()

    def publish_events(self, events: List[base_types.DomainEvent]) -> None:
        self._events_to_publish.extend(events)
