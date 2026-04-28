import pytest
import boto3
from boto3.dynamodb.conditions import Attr, Key
from botocore.exceptions import ClientError
from moto import mock_aws

from erns_shared.aws.dynamodb import DynamoDBTable

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
        ddb.put_item({"pk": "user#1", "sk": "profile", "name": "Alice", "active": True})
        ddb.put_item({"pk": "user#1", "sk": "order#001", "total": 99, "status": "shipped"})
        ddb.put_item({"pk": "user#1", "sk": "order#002", "total": 42, "status": "pending"})
        ddb.put_item({"pk": "user#1", "sk": "order#003", "total": 75, "status": "shipped"})
        ddb.put_item({"pk": "user#2", "sk": "profile", "name": "Bob", "active": False})
        yield ddb


# ---------------------------------------------------------------------------
# query_by_pk — sk_condition
# ---------------------------------------------------------------------------


class TestQueryByPkSkCondition:
    def test_no_sk_condition_returns_all(self, table):
        assert len(list(table.query_by_pk(pk_name=_PK, pk_value="user#1"))) == 4

    def test_sk_eq(self, table):
        items = list(table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("profile")))
        assert len(items) == 1
        assert items[0]["name"] == "Alice"

    def test_sk_begins_with(self, table):
        items = list(table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).begins_with("order#")))
        assert len(items) == 3

    def test_sk_between(self, table):
        items = list(
            table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).between("order#001", "order#002"))
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
            table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).begins_with("order#"), scan_index_forward=False)
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
                _PK, "user#1", _SK, "order#",
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
            table.scan(filter_expression=Attr("status").eq("shipped") & Attr("total").gte(80))
        )
        assert len(items) == 1
        assert items[0]["total"] == 99


# ---------------------------------------------------------------------------
# put_item — condition + return_values
# ---------------------------------------------------------------------------


class TestPutItem:
    def test_returns_none_by_default(self, table):
        assert table.put_item({"pk": "new#1", "sk": "x"}) is None

    def test_return_all_old_gives_previous_item(self, table):
        old = table.put_item(
            {"pk": "user#1", "sk": "profile", "name": "Alice Updated"},
            return_values="ALL_OLD",
        )
        assert old is not None
        assert old["name"] == "Alice"

    def test_return_all_old_with_no_previous_item_is_none(self, table):
        assert table.put_item({"pk": "brand#new", "sk": "x"}, return_values="ALL_OLD") is None

    def test_condition_passes_and_writes(self, table):
        table.put_item(
            {"pk": "user#1", "sk": "profile", "name": "Alice v2"},
            condition=Attr("name").eq("Alice"),
        )
        items = list(table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("profile")))
        assert items[0]["name"] == "Alice v2"

    def test_condition_fails_raises(self, table):
        with pytest.raises(ClientError, match="ConditionalCheckFailedException"):
            table.put_item(
                {"pk": "user#1", "sk": "profile", "name": "X"},
                condition=Attr("name").eq("NotAlice"),
            )

    def test_attribute_not_exists_prevents_overwrite(self, table):
        with pytest.raises(ClientError, match="ConditionalCheckFailedException"):
            table.put_item(
                {"pk": "user#1", "sk": "profile"},
                condition=Attr("pk").not_exists(),
            )


# ---------------------------------------------------------------------------
# delete_item — condition
# ---------------------------------------------------------------------------


class TestDeleteItem:
    def test_simple_delete(self, table):
        table.delete_item({"pk": "user#1", "sk": "order#001"})
        assert list(table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#001"))) == []

    def test_condition_passes(self, table):
        table.delete_item(
            {"pk": "user#1", "sk": "order#001"},
            condition=Attr("status").eq("shipped"),
        )
        assert list(table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#001"))) == []

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
            w.put({"pk": "user#3", "sk": "profile", "name": "Carol"})
            w.put({"pk": "user#4", "sk": "profile", "name": "Dave"})

        assert list(table.query_by_pk(_PK, "user#3"))[0]["name"] == "Carol"
        assert list(table.query_by_pk(_PK, "user#4"))[0]["name"] == "Dave"

    def test_deletes_on_exit(self, table):
        with table.batch_writer() as w:
            w.delete({"pk": "user#1", "sk": "order#001"})

        remaining = list(table.query_by_pk_sk_prefix(_PK, "user#1", _SK, "order#"))
        assert len(remaining) == 2

    def test_mixed_put_and_delete(self, table):
        with table.batch_writer() as w:
            w.put({"pk": "user#7", "sk": "profile", "name": "Grace"})
            w.delete({"pk": "user#1", "sk": "order#001"})

        assert list(table.query_by_pk(_PK, "user#7"))[0]["name"] == "Grace"
        assert list(table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#001"))) == []

    def test_condition_raises_immediately(self, table):
        with pytest.raises(ValueError, match="batch mode"):
            with table.batch_writer() as w:
                w.put({"pk": "x", "sk": "y"}, condition=Attr("pk").not_exists())

    def test_exception_rolls_back_pending_ops(self, table):
        with pytest.raises(RuntimeError):
            with table.batch_writer() as w:
                w.put({"pk": "ghost#1", "sk": "x"})
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
            w.put({"pk": "user#5", "sk": "profile", "name": "Eve"})
            w.put({"pk": "user#6", "sk": "profile", "name": "Frank"})

        assert list(table.query_by_pk(_PK, "user#5"))[0]["name"] == "Eve"
        assert list(table.query_by_pk(_PK, "user#6"))[0]["name"] == "Frank"

    def test_mixed_put_and_delete_atomically(self, table):
        with table.transaction_writer() as w:
            w.put({"pk": "user#8", "sk": "profile", "name": "Heidi"})
            w.delete({"pk": "user#1", "sk": "order#003"})

        assert list(table.query_by_pk(_PK, "user#8"))[0]["name"] == "Heidi"
        assert list(table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#003"))) == []

    def test_condition_on_put(self, table):
        with table.transaction_writer() as w:
            w.put(
                {"pk": "user#1", "sk": "profile", "name": "Alice v2"},
                condition=Attr("name").eq("Alice"),
            )
        assert list(table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("profile")))[0]["name"] == "Alice v2"

    def test_condition_on_delete(self, table):
        with table.transaction_writer() as w:
            w.delete(
                {"pk": "user#1", "sk": "order#001"},
                condition=Attr("status").eq("shipped"),
            )
        assert list(table.query_by_pk(_PK, "user#1", sk_condition=Key(_SK).eq("order#001"))) == []

    def test_exceeding_100_ops_raises_before_flush(self, table):
        with pytest.raises(ValueError, match="100"):
            with table.transaction_writer() as w:
                for i in range(101):
                    w.put({"pk": f"x#{i}", "sk": "y"})

    def test_exception_rolls_back_pending_ops(self, table):
        with pytest.raises(RuntimeError):
            with table.transaction_writer() as w:
                w.put({"pk": "ghost#2", "sk": "x"})
                raise RuntimeError("abort")

        assert list(table.query_by_pk(_PK, "ghost#2")) == []
