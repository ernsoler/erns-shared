from typing import Optional, Iterator, Type, Dict, Any, Final
from erns_shared.ddd.adapters.persistence import commons
from erns_shared.ddd.adapters.persistence.commons import E, I

MAX_DYNAMO_DB_BATCH_SIZE_PER_TRX: Final = 100


class DynamoDbRepository(commons.Repository[E]):
    def __init__(
        self,
        session: commons.SessionDB,
        table_name: str,
        entity_type: Type[E],
    ) -> None:
        self._session = session
        self._table_name = table_name
        self._key_name: Final = "id._key"
        self._entity_type = entity_type

    @classmethod
    def _deserialize_item(cls, dynamodb_record: Dict[str, Any]) -> Dict[str, Any]:
        from boto3.dynamodb.types import TypeDeserializer

        deserializer = TypeDeserializer()
        return {k: deserializer.deserialize(v) for k, v in dynamodb_record.items()}

    @classmethod
    def _serialize_entity(cls, entity: E) -> Dict[str, Any]:
        from decimal import Decimal

        nested_dict = entity.model_dump()
        for k, v in nested_dict.items():
            if isinstance(v, Decimal):
                nested_dict[k] = float(v)
        return nested_dict

    def put(self, item: E) -> None:
        self._session.add_write_operation(
            operation=self._build_write_operation(
                item=item, operation_type=_DynamoDbPutOperation
            )
        )

    def update(self, item: E) -> None:
        self._session.add_write_operation(
            operation=self._build_write_operation(
                item=item, operation_type=_DynamoDbUpdateOperation
            )
        )

    def _build_write_operation(
        self,
        item: E,
        operation_type: Type["_DynamoDbWriteOperation"],
    ) -> "_DynamoDbWriteOperation":
        item_to_save = item.model_copy()
        item_to_save._increase_version()
        record_serialized = self._serialize_entity(item_to_save)
        record_serialized[self._key_name] = item_to_save.id._key()
        return operation_type(
            table_name=self._table_name,
            key_name=self._key_name,
            entity_dict=record_serialized,
        )

    def get_by_id(self, id: I) -> E:
        item = self._session.client.get_item(
            TableName=self._table_name, Key={self._key_name: {"S": id._key()}}
        ).get("Item")
        if item:
            return self._entity_type.model_validate(
                self._deserialize_item(dynamodb_record=item)
            )
        raise ValueError("Requested item not found")

    def find_by_id(self, id: I) -> Optional[E]:
        try:
            return self.get_by_id(id=id)
        except ValueError:
            return None

    def get_all(self) -> Iterator[E]:
        params: Dict[str, Any] = {
            "TableName": self._table_name,
            "Limit": MAX_DYNAMO_DB_BATCH_SIZE_PER_TRX,
        }
        while True:
            response = self._session.client.scan(**params)
            for item in response.get("Items", []):
                yield self._entity_type.model_validate(
                    self._deserialize_item(dynamodb_record=item)
                )

            if "LastEvaluatedKey" in response:
                params["ExclusiveStartKey"] = response["LastEvaluatedKey"]
            else:
                break


class _DynamoDbWriteOperation:
    def __init__(
        self, table_name: str, key_name: str, entity_dict: Dict[str, Any]
    ) -> None:
        self._table_name = table_name
        self._key_name = key_name
        self._id = entity_dict.get(key_name)
        self._entity = entity_dict

    @property
    def id(self) -> str:
        return self._id  # type: ignore

    @property
    def key_name(self) -> str:
        return self._key_name

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def entity_serialized(self) -> Dict[str, Any]:
        from boto3.dynamodb.types import TypeSerializer

        serializer = TypeSerializer()
        return {k: serializer.serialize(v) for k, v in self._entity.items()}


class _DynamoDbPutOperation(_DynamoDbWriteOperation):
    @property
    def entity_serialized(self) -> Dict[str, Any]:
        entity_dict = super().entity_serialized
        return {
            "Put": {
                "Item": entity_dict,
                "TableName": self.table_name,
                "ConditionExpression": "attribute_not_exists(#id)",
                "ExpressionAttributeNames": {"#id": self.key_name},
                "ReturnValuesOnConditionCheckFailure": "ALL_OLD",
            }
        }


class _DynamoDbUpdateOperation(_DynamoDbWriteOperation):
    @property
    def entity_serialized(self) -> Dict[str, Any]:
        entity_dict = super().entity_serialized
        key_value = entity_dict.pop(self.key_name)
        expression_attr_names: Dict[str, str] = {}
        expression_attr_values: Dict[str, Any] = {}
        update_parts = []
        for attr_name, attr_value in entity_dict.items():
            expression_attr_names[f"#{attr_name}"] = attr_name
            expression_attr_values[f":{attr_name}"] = attr_value
            update_parts.append(f"#{attr_name} = :{attr_name}")
        return {
            "Update": {
                "TableName": self.table_name,
                "Key": {self.key_name: key_value},
                "UpdateExpression": "SET " + ", ".join(update_parts),
                "ExpressionAttributeNames": expression_attr_names,
                "ExpressionAttributeValues": expression_attr_values,
            }
        }
