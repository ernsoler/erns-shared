import pytest
import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError
from moto import mock_aws

from erns_shared.aws.dynamodb import DynamoDBTable, DynamoDbId, DynamoDbRecord


class _Item(DynamoDbRecord):
    name: str
    active: bool = True


class _Order(DynamoDbRecord):
    total: int
    status: str


_TABLE = "test-table"
_PK = "pk"
_SK = "sk"


@pytest.fixture
def table(aws_credentials):
    with mock_aws():
        boto3.client("dynamodb", region_name="us-east-1").create_table(
            TableName=_TABLE,
            KeySchema=[
                {"AttributeName": _PK, "KeyType": "HASH"},
                {"AttributeName": _SK, "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": _PK, "AttributeType": "S"},
                {"AttributeName": _SK, "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        ddb = DynamoDBTable(_TABLE)
        ddb.put_item(
            _Item(id=DynamoDbId(pk="user#1", sk="profile"), name="Alice", active=True)
        )
        ddb.put_item(
            _Order(
                id=DynamoDbId(pk="user#1", sk="order#001"), total=99, status="shipped"
            )
        )
        ddb.put_item(
            _Order(
                id=DynamoDbId(pk="user#1", sk="order#002"), total=42, status="pending"
            )
        )
        ddb.put_item(
            _Order(
                id=DynamoDbId(pk="user#1", sk="order#003"), total=75, status="shipped"
            )
        )
        ddb.put_item(
            _Item(id=DynamoDbId(pk="user#2", sk="profile"), name="Bob", active=False)
        )
        yield ddb


# ---------------------------------------------------------------------------
# query_by_pk — sk_condition
# ---------------------------------------------------------------------------


class TestQueryByPkSkCondition:
    def test_no_sk_condition_returns_all(self, table):
        assert len(list(table.query_by_pk(pk_name=_PK, pk_value="user#1"))) == 4

    def test_sk_eq(self, table):
        items = list(
            table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("profile"))
        )
        assert len(items) == 1
        assert items[0]["name"] == "Alice"

    def test_sk_begins_with(self, table):
        items = list(
            table.query_by_pk(
                _PK, "user#1", sk_condition=Key(_SK).begins_with("order#")
            )
        )
        assert len(items) == 3

    def test_sk_between(self, table):
        items = list(
            table.query_by_pk(
                _PK, "user#1", sk_condition=Key(_SK).between("order#001", "order#002")
            )
        )
        assert len(items) == 2
        assert {i["sk"] for i in items} == {"order#001", "order#002"}

    def test_sk_condition_and_filter_expression(self, table):
        items = list(
            table.query_by_pk(
                _PK,
                "user#1",
                sk_condition=Key(_SK).begins_with("order#"),
                filter_expression=Attr("status").eq("shipped"),
            )
        )
        assert len(items) == 2
        assert all(i["status"] == "shipped" for i in items)

    def test_scan_index_forward_false_reverses_order(self, table):
        items = list(
            table.query_by_pk(
                _PK,
                "user#1",
                sk_condition=Key(_SK).begins_with("order#"),
                scan_index_forward=False,
            )
        )
        sks = [i["sk"] for i in items]
        assert sks == sorted(sks, reverse=True)

    def test_unknown_pk_returns_empty(self, table):
        assert list(table.query_by_pk(pk_name=_PK, pk_value="user#99")) == []


# ---------------------------------------------------------------------------
# query_by_pk_sk_prefix
# ---------------------------------------------------------------------------


class TestQueryByPkSkPrefix:
    def test_filters_by_prefix(self, table):
        items = list(table.query_by_pk_sk_prefix(_PK, "user#1", _SK, "order#"))
        assert len(items) == 3

    def test_filter_expression_stacks(self, table):
        items = list(
            table.query_by_pk_sk_prefix(
                _PK,
                "user#1",
                _SK,
                "order#",
                filter_expression=Attr("status").eq("pending"),
            )
        )
        assert len(items) == 1
        assert items[0]["sk"] == "order#002"

    def test_no_match_returns_empty(self, table):
        assert list(table.query_by_pk_sk_prefix(_PK, "user#1", _SK, "invoice#")) == []


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


class TestScan:
    def test_returns_all_items(self, table):
        assert len(list(table.scan())) == 5

    def test_filter_by_attr_value(self, table):
        items = list(table.scan(filter_expression=Attr("active").eq(True)))
        assert len(items) == 1
        assert items[0]["name"] == "Alice"

    def test_filter_by_attr_exists(self, table):
        items = list(table.scan(filter_expression=Attr("status").exists()))
        assert len(items) == 3

    def test_compound_filter(self, table):
        items = list(
            table.scan(
                filter_expression=Attr("status").eq("shipped") & Attr("total").gte(80)
            )
        )
        assert len(items) == 1
        assert items[0]["total"] == 99


# ---------------------------------------------------------------------------
# put_item — condition + return_values
# ---------------------------------------------------------------------------


class TestPutItem:
    def test_returns_none_by_default(self, table):
        assert (
            table.put_item(_Item(id=DynamoDbId(pk="new#1", sk="x"), name="New")) is None
        )

    def test_return_all_old_gives_previous_item(self, table):
        old = table.put_item(
            _Item(id=DynamoDbId(pk="user#1", sk="profile"), name="Alice Updated"),
            return_values="ALL_OLD",
        )
        assert old is not None
        assert old["name"] == "Alice"

    def test_return_all_old_with_no_previous_item_is_none(self, table):
        assert (
            table.put_item(
                _Item(id=DynamoDbId(pk="brand#new", sk="x"), name="New"),
                return_values="ALL_OLD",
            )
            is None
        )

    def test_condition_passes_and_writes(self, table):
        table.put_item(
            _Item(id=DynamoDbId(pk="user#1", sk="profile"), name="Alice v2"),
            condition=Attr("name").eq("Alice"),
        )
        items = list(
            table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("profile"))
        )
        assert items[0]["name"] == "Alice v2"

    def test_condition_fails_raises(self, table):
        with pytest.raises(ClientError, match="ConditionalCheckFailedException"):
            table.put_item(
                _Item(id=DynamoDbId(pk="user#1", sk="profile"), name="X"),
                condition=Attr("name").eq("NotAlice"),
            )

    def test_attribute_not_exists_prevents_overwrite(self, table):
        with pytest.raises(ClientError, match="ConditionalCheckFailedException"):
            table.put_item(
                _Item(id=DynamoDbId(pk="user#1", sk="profile"), name="Alice"),
                condition=Attr(_PK).not_exists(),
            )


# ---------------------------------------------------------------------------
# delete_item — condition
# ---------------------------------------------------------------------------


class TestDeleteItem:
    def test_simple_delete(self, table):
        table.delete_item({"pk": "user#1", "sk": "order#001"})
        assert (
            list(
                table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#001"))
            )
            == []
        )

    def test_condition_passes(self, table):
        table.delete_item(
            {"pk": "user#1", "sk": "order#001"},
            condition=Attr("status").eq("shipped"),
        )
        assert (
            list(
                table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#001"))
            )
            == []
        )

    def test_condition_fails_raises(self, table):
        with pytest.raises(ClientError, match="ConditionalCheckFailedException"):
            table.delete_item(
                {"pk": "user#1", "sk": "order#001"},
                condition=Attr("status").eq("pending"),
            )


# ---------------------------------------------------------------------------
# batch_get
# ---------------------------------------------------------------------------


class TestBatchGet:
    def test_returns_requested_items(self, table):
        keys = [{"pk": "user#1", "sk": "profile"}, {"pk": "user#2", "sk": "profile"}]
        items = list(table.batch_get(keys=keys))
        assert {i["name"] for i in items} == {"Alice", "Bob"}

    def test_missing_key_not_returned(self, table):
        keys = [{"pk": "user#1", "sk": "profile"}, {"pk": "ghost#99", "sk": "profile"}]
        items = list(table.batch_get(keys=keys))
        assert len(items) == 1


# ---------------------------------------------------------------------------
# batch_writer
# ---------------------------------------------------------------------------


class TestBatchWriter:
    def test_puts_on_exit(self, table):
        with table.batch_writer() as w:
            w.put(_Item(id=DynamoDbId(pk="user#3", sk="profile"), name="Carol"))
            w.put(_Item(id=DynamoDbId(pk="user#4", sk="profile"), name="Dave"))

        assert list(table.query_by_pk(_PK, "user#3"))[0]["name"] == "Carol"
        assert list(table.query_by_pk(_PK, "user#4"))[0]["name"] == "Dave"

    def test_deletes_on_exit(self, table):
        with table.batch_writer() as w:
            w.delete(DynamoDbId(pk="user#1", sk="order#001"))

        remaining = list(table.query_by_pk_sk_prefix(_PK, "user#1", _SK, "order#"))
        assert len(remaining) == 2

    def test_mixed_put_and_delete(self, table):
        with table.batch_writer() as w:
            w.put(_Item(id=DynamoDbId(pk="user#7", sk="profile"), name="Grace"))
            w.delete(DynamoDbId(pk="user#1", sk="order#001"))

        assert list(table.query_by_pk(_PK, "user#7"))[0]["name"] == "Grace"
        assert (
            list(
                table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#001"))
            )
            == []
        )

    def test_condition_raises_immediately(self, table):
        with pytest.raises(ValueError, match="batch mode"):
            with table.batch_writer() as w:
                w.put(
                    _Item(id=DynamoDbId(pk="x", sk="y"), name="X"),
                    condition=Attr("pk").not_exists(),
                )

    def test_exception_rolls_back_pending_ops(self, table):
        with pytest.raises(RuntimeError):
            with table.batch_writer() as w:
                w.put(_Item(id=DynamoDbId(pk="ghost#1", sk="x"), name="Ghost"))
                raise RuntimeError("abort")

        assert list(table.query_by_pk(_PK, "ghost#1")) == []

    def test_empty_writer_is_noop(self, table):
        before = len(list(table.scan()))
        with table.batch_writer():
            pass
        assert len(list(table.scan())) == before


# ---------------------------------------------------------------------------
# transaction_writer
# ---------------------------------------------------------------------------


class TestTransactionWriter:
    def test_puts_atomically(self, table):
        with table.transaction_writer() as w:
            w.put(_Item(id=DynamoDbId(pk="user#5", sk="profile"), name="Eve"))
            w.put(_Item(id=DynamoDbId(pk="user#6", sk="profile"), name="Frank"))

        assert list(table.query_by_pk(_PK, "user#5"))[0]["name"] == "Eve"
        assert list(table.query_by_pk(_PK, "user#6"))[0]["name"] == "Frank"

    def test_mixed_put_and_delete_atomically(self, table):
        with table.transaction_writer() as w:
            w.put(_Item(id=DynamoDbId(pk="user#8", sk="profile"), name="Heidi"))
            w.delete(DynamoDbId(pk="user#1", sk="order#003"))

        assert list(table.query_by_pk(_PK, "user#8"))[0]["name"] == "Heidi"
        assert (
            list(
                table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#003"))
            )
            == []
        )

    def test_condition_on_put(self, table):
        with table.transaction_writer() as w:
            w.put(
                _Item(id=DynamoDbId(pk="user#1", sk="profile"), name="Alice v2"),
                condition=Attr("name").eq("Alice"),
            )
        assert (
            list(table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("profile")))[
                0
            ]["name"]
            == "Alice v2"
        )

    def test_condition_on_delete(self, table):
        with table.transaction_writer() as w:
            w.delete(
                DynamoDbId(pk="user#1", sk="order#001"),
                condition=Attr("status").eq("shipped"),
            )
        assert (
            list(
                table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#001"))
            )
            == []
        )

    def test_exceeding_100_ops_raises_before_flush(self, table):
        with pytest.raises(ValueError, match="100"):
            with table.transaction_writer() as w:
                for i in range(101):
                    w.put(_Item(id=DynamoDbId(pk=f"x#{i}", sk="y"), name="X"))

    def test_exception_rolls_back_pending_ops(self, table):
        with pytest.raises(RuntimeError):
            with table.transaction_writer() as w:
                w.put(_Item(id=DynamoDbId(pk="ghost#2", sk="x"), name="Ghost"))
                raise RuntimeError("abort")

        assert list(table.query_by_pk(_PK, "ghost#2")) == []


# ---------------------------------------------------------------------------
# DynamoDbId
# ---------------------------------------------------------------------------


class TestDynamoDbId:
    def test_pk_and_sk(self):
        id = DynamoDbId(pk="user#1", sk="profile")
        assert id.pk == "user#1"
        assert id.sk == "profile"

    def test_sk_is_optional(self):
        id = DynamoDbId(pk="user#1")
        assert id.sk is None

    def test_is_frozen(self):
        id = DynamoDbId(pk="user#1", sk="profile")
        with pytest.raises(Exception):
            id.pk = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DynamoDbRecord
# ---------------------------------------------------------------------------


class TestDynamoDbRecord:
    def test_requires_id(self):
        with pytest.raises(Exception):
            _Item(name="Alice")  # type: ignore[call-arg]

    def test_top_level_pk_sk_ignored_on_validate(self):
        # DynamoDB items have pk/sk hoisted to top level — they must be ignored
        record = _Item.model_validate(
            {
                "id": {"pk": "user#1", "sk": "profile"},
                "pk": "user#1",
                "sk": "profile",
                "name": "Alice",
            }
        )
        assert record.id.pk == "user#1"
        assert record.name == "Alice"


# ---------------------------------------------------------------------------
# DynamoDBTable helpers
# ---------------------------------------------------------------------------


class TestGetKey:
    def test_with_sk(self, table):
        assert table._get_key(DynamoDbId(pk="user#1", sk="profile")) == {
            "pk": "user#1",
            "sk": "profile",
        }

    def test_without_sk(self, table):
        assert table._get_key(DynamoDbId(pk="user#1")) == {"pk": "user#1"}

    def test_custom_key_names(self, aws_credentials):
        with mock_aws():
            boto3.client("dynamodb", region_name="us-east-1").create_table(
                TableName="other",
                KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "PK", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
            t = DynamoDBTable("other", pk_name="PK", sk_name="SK")
            assert t._get_key(DynamoDbId(pk="x", sk="y")) == {"PK": "x", "SK": "y"}
            assert t._get_key(DynamoDbId(pk="x")) == {"PK": "x"}


class TestSerialize:
    def test_injects_pk_and_sk(self, table):
        record = _Item(id=DynamoDbId(pk="user#1", sk="profile"), name="Alice")
        item = table._serialize(record)
        assert item["pk"] == "user#1"
        assert item["sk"] == "profile"
        assert item["name"] == "Alice"

    def test_keeps_nested_id(self, table):
        record = _Item(id=DynamoDbId(pk="user#1", sk="profile"), name="Alice")
        item = table._serialize(record)
        assert item["id"] == {"pk": "user#1", "sk": "profile"}

    def test_decimal_converted_to_float(self, table):
        from decimal import Decimal

        class _Scored(DynamoDbRecord):
            score: Decimal

        record = _Scored(id=DynamoDbId(pk="u#1", sk="s"), score=Decimal("9.5"))
        item = table._serialize(record)
        assert item["score"] == 9.5
        assert isinstance(item["score"], float)

    def test_sk_omitted_when_none(self, table):
        record = _Item(id=DynamoDbId(pk="user#1"), name="Alice")
        item = table._serialize(record)
        assert "pk" in item
        assert "sk" not in item


class TestDeserialize:
    def test_returns_typed_record(self, table):
        item = {
            "id": {"pk": "user#1", "sk": "profile"},
            "pk": "user#1",
            "sk": "profile",
            "name": "Alice",
        }
        record = table._deserialize(item, _Item)
        assert isinstance(record, _Item)
        assert record.id.pk == "user#1"
        assert record.name == "Alice"

    def test_extra_keys_ignored(self, table):
        item = {"id": {"pk": "u#1", "sk": "s"}, "name": "Bob", "unknown_field": "x"}
        record = table._deserialize(item, _Item)
        assert record.name == "Bob"


# ---------------------------------------------------------------------------
# update_item_by_id
# ---------------------------------------------------------------------------


class TestUpdateItemById:
    def test_updates_attribute(self, table):
        table.update_item_by_id(
            DynamoDbId(pk="user#1", sk="profile"), {"name": "Alice Updated"}
        )
        item = list(
            table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("profile"))
        )[0]
        assert item["name"] == "Alice Updated"

    def test_updates_multiple_attributes(self, table):
        table.update_item_by_id(
            DynamoDbId(pk="user#1", sk="order#001"),
            {"status": "cancelled", "total": 0},
        )
        item = list(
            table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#001"))
        )[0]
        assert item["status"] == "cancelled"
        assert item["total"] == 0

    def test_empty_updates_is_noop(self, table):
        before = list(
            table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("profile"))
        )[0]
        table.update_item_by_id(DynamoDbId(pk="user#1", sk="profile"), {})
        after = list(
            table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("profile"))
        )[0]
        assert before == after

    def test_condition_passes(self, table):
        table.update_item_by_id(
            DynamoDbId(pk="user#1", sk="profile"),
            {"name": "Alice v2"},
            condition=Attr("name").eq("Alice"),
        )
        item = list(
            table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("profile"))
        )[0]
        assert item["name"] == "Alice v2"

    def test_condition_fails_raises(self, table):
        with pytest.raises(ClientError, match="ConditionalCheckFailedException"):
            table.update_item_by_id(
                DynamoDbId(pk="user#1", sk="profile"),
                {"name": "X"},
                condition=Attr("name").eq("NotAlice"),
            )
