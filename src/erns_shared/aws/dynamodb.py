import contextlib
from dataclasses import dataclass, field
import boto3
from boto3.dynamodb.conditions import ConditionBase, ConditionExpressionBuilder, Key as DynamoKey
from boto3.dynamodb.types import TypeSerializer
from typing import Any, Dict, Iterator, List, Literal, Optional

from erns_shared.ddd.base_types import split_list

_BATCH_WRITE_LIMIT = 25
_BATCH_GET_LIMIT = 100
_TRANSACTION_LIMIT = 100


class DynamoDBTable:
    """High-level single-table query helpers. Complements the DDD persistence layer."""

    def __init__(self, table_name: str) -> None:
        self._table_name = table_name
        self._table = boto3.resource("dynamodb").Table(table_name)

    # ---- queries -----------------------------------------------------------

    def query_by_pk(
        self,
        pk_name: str,
        pk_value: str,
        sk_condition: Optional[ConditionBase] = None,
        index_name: Optional[str] = None,
        scan_index_forward: bool = True,
        filter_expression: Optional[ConditionBase] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Query by partition key with an optional sort key condition.

        sk_condition examples:
            Key("sk").eq("profile")
            Key("sk").begins_with("order#")
            Key("sk").between("order#2024", "order#2025")
            Key("sk").gt("order#100")
        """
        key_cond = DynamoKey(pk_name).eq(pk_value)
        if sk_condition is not None:
            key_cond = key_cond & sk_condition
        params: Dict[str, Any] = {
            "KeyConditionExpression": key_cond,
            "ScanIndexForward": scan_index_forward,
        }
        if index_name:
            params["IndexName"] = index_name
        if filter_expression is not None:
            params["FilterExpression"] = filter_expression
        yield from self._paginate_query(params)

    def query_by_pk_sk_prefix(
        self,
        pk_name: str,
        pk_value: str,
        sk_name: str,
        sk_prefix: str,
        index_name: Optional[str] = None,
        scan_index_forward: bool = True,
        filter_expression: Optional[ConditionBase] = None,
    ) -> Iterator[Dict[str, Any]]:
        yield from self.query_by_pk(
            pk_name=pk_name,
            pk_value=pk_value,
            sk_condition=DynamoKey(sk_name).begins_with(sk_prefix),
            index_name=index_name,
            scan_index_forward=scan_index_forward,
            filter_expression=filter_expression,
        )

    def scan(
        self,
        filter_expression: Optional[ConditionBase] = None,
        index_name: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if filter_expression is not None:
            params["FilterExpression"] = filter_expression
        if index_name:
            params["IndexName"] = index_name
        while True:
            response = self._table.scan(**params)
            yield from response.get("Items", [])
            if "LastEvaluatedKey" not in response:
                break
            params["ExclusiveStartKey"] = response["LastEvaluatedKey"]

    def batch_get(self, keys: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        resource = boto3.resource("dynamodb")
        for chunk in split_list(keys, _BATCH_GET_LIMIT):
            response = resource.batch_get_item(
                RequestItems={self._table_name: {"Keys": chunk}}
            )
            yield from response["Responses"].get(self._table_name, [])
            if response.get("UnprocessedKeys"):
                raise RuntimeError(
                    f"DynamoDB returned unprocessed keys for table {self._table_name}"
                )

    def put_item(
        self,
        item: Dict[str, Any],
        condition: Optional[ConditionBase] = None,
        return_values: Literal["NONE", "ALL_OLD"] = "NONE",
    ) -> Optional[Dict[str, Any]]:
        """Write an item. Returns the previous item when return_values="ALL_OLD"."""
        kwargs: Dict[str, Any] = {"Item": item, "ReturnValues": return_values}
        if condition is not None:
            kwargs["ConditionExpression"] = condition
        response = self._table.put_item(**kwargs)
        return response.get("Attributes") or None

    def delete_item(
        self,
        key: Dict[str, Any],
        condition: Optional[ConditionBase] = None,
    ) -> None:
        kwargs: Dict[str, Any] = {"Key": key}
        if condition is not None:
            kwargs["ConditionExpression"] = condition
        self._table.delete_item(**kwargs)

    def batch_writer(self) -> contextlib.AbstractContextManager["_WriteContext"]:
        """Accumulate puts/deletes and flush via batch_write_item on exit.

        Not atomic — AWS limit is 25 per call, so large batches are chunked
        into multiple calls automatically. ConditionExpression not supported.
        """
        return self._writer("batch")

    def transaction_writer(self) -> contextlib.AbstractContextManager["_WriteContext"]:
        """Accumulate puts/deletes and flush via transact_write_items on exit.

        Fully atomic — either all operations land or none do.
        Supports ConditionExpression per operation.
        Raises ValueError if more than 100 operations are queued (AWS hard limit).
        """
        return self._writer("transaction")

    @contextlib.contextmanager
    def _writer(self, mode: Literal["batch", "transaction"]) -> Iterator["_WriteContext"]:
        ctx = _WriteContext(table_name=self._table_name, mode=mode)
        try:
            yield ctx
            ctx._flush()
        except Exception:
            ctx._ops.clear()
            raise

    def _paginate_query(self, params: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        while True:
            response = self._table.query(**params)
            yield from response.get("Items", [])
            if "LastEvaluatedKey" not in response:
                break
            params["ExclusiveStartKey"] = response["LastEvaluatedKey"]


@dataclass
class _Op:
    kind: Literal["put", "delete"]
    data: Dict[str, Any]
    condition: Optional[ConditionBase] = field(default=None)


class _WriteContext:
    def __init__(self, table_name: str, mode: Literal["batch", "transaction"]) -> None:
        self._table_name = table_name
        self._mode = mode
        self._ops: List[_Op] = []

    def put(self, item: Dict[str, Any], condition: Optional[ConditionBase] = None) -> None:
        if condition is not None and self._mode == "batch":
            raise ValueError(
                "ConditionExpression is not supported in batch mode — use transaction_writer()"
            )
        self._ops.append(_Op(kind="put", data=item, condition=condition))

    def delete(self, key: Dict[str, Any], condition: Optional[ConditionBase] = None) -> None:
        if condition is not None and self._mode == "batch":
            raise ValueError(
                "ConditionExpression is not supported in batch mode — use transaction_writer()"
            )
        self._ops.append(_Op(kind="delete", data=key, condition=condition))

    def _flush(self) -> None:
        if not self._ops:
            return
        if self._mode == "batch":
            self._flush_batch()
        else:
            self._flush_transaction()

    def _flush_batch(self) -> None:
        resource = boto3.resource("dynamodb")
        for chunk in split_list(self._ops, _BATCH_WRITE_LIMIT):
            requests = [
                {"PutRequest": {"Item": op.data}} if op.kind == "put"
                else {"DeleteRequest": {"Key": op.data}}
                for op in chunk
            ]
            resource.batch_write_item(
                RequestItems={self._table_name: requests})

    def _flush_transaction(self) -> None:
        if len(self._ops) > _TRANSACTION_LIMIT:
            raise ValueError(
                f"Transaction mode supports at most {_TRANSACTION_LIMIT} operations, "
                f"got {len(self._ops)}"
            )
        serializer = TypeSerializer()
        transact_items = []
        for op in self._ops:
            serialized = {k: serializer.serialize(
                v) for k, v in op.data.items()}
            entry: Dict[str, Any]
            if op.kind == "put":
                entry = {
                    "Put": {"TableName": self._table_name, "Item": serialized}}
            else:
                entry = {"Delete": {
                    "TableName": self._table_name, "Key": serialized}}
            if op.condition is not None:
                entry[op.kind.capitalize()].update(
                    _serialize_condition(op.condition))
            transact_items.append(entry)
        boto3.client("dynamodb").transact_write_items(
            TransactItems=transact_items)


def _serialize_condition(condition: ConditionBase) -> Dict[str, Any]:
    """Translate a boto3 ConditionBase into the dict shape expected by the low-level client."""
    built = ConditionExpressionBuilder().build_expression(condition)
    result: Dict[str, Any] = {
        "ConditionExpression": built.condition_expression}
    if built.attribute_name_placeholders:
        result["ExpressionAttributeNames"] = dict(
            built.attribute_name_placeholders)
    if built.attribute_value_placeholders:
        serializer = TypeSerializer()
        result["ExpressionAttributeValues"] = {
            k: serializer.serialize(v)
            for k, v in built.attribute_value_placeholders.items()
        }
    return result
