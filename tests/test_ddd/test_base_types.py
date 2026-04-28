import enum
import time
import pytest
from erns_shared.ddd.base_types import (
    DomainAggregate,
    DomainEvent,
    EntityId,
    EpochTime,
    Inmutable,
    NamedEnum,
    RootEntity,
    UUIDGenerator,
    split_list,
    update_last_update_date,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _SampleId(EntityId):
    value: str


class _SampleEntity(RootEntity):
    id: _SampleId
    name: str


class _SampleAggregate(DomainAggregate):
    id: _SampleId
    name: str


class _SampleEvent(DomainEvent):
    domain_name: str = "test_domain"
    payload: str


# ---------------------------------------------------------------------------
# NamedEnum
# ---------------------------------------------------------------------------


class TestNamedEnum:
    def test_value_equals_name(self):
        class Color(NamedEnum):
            RED = enum.auto()
            BLUE = enum.auto()

        assert Color.RED.value == "RED"
        assert Color.BLUE.value == "BLUE"
        assert str(Color.RED) == "RED"


# ---------------------------------------------------------------------------
# UUIDGenerator
# ---------------------------------------------------------------------------


class TestUUIDGenerator:
    def test_returns_string(self):
        assert isinstance(UUIDGenerator.uuid(), str)

    def test_values_are_unique(self):
        ids = {UUIDGenerator.uuid() for _ in range(200)}
        assert len(ids) == 200


# ---------------------------------------------------------------------------
# EpochTime
# ---------------------------------------------------------------------------


class TestEpochTime:
    def test_now_is_positive_int(self):
        epoch = EpochTime.now()
        assert isinstance(epoch.time_ns, int)
        assert epoch.time_ns > 0

    def test_str_representation(self):
        assert str(EpochTime(time_ns=99)) == "99"

    def test_is_frozen(self):
        epoch = EpochTime.now()
        with pytest.raises(Exception):
            epoch.time_ns = 0  # type: ignore

    def test_now_is_monotonically_increasing(self):
        t1 = EpochTime.now()
        time.sleep(0.001)
        t2 = EpochTime.now()
        assert t2.time_ns > t1.time_ns


# ---------------------------------------------------------------------------
# EntityId / Key
# ---------------------------------------------------------------------------


class TestEntityId:
    def test_single_field_key(self):
        eid = _SampleId(value="abc-123")
        assert eid._key() == "abc-123"

    def test_multi_field_key_joined_by_hash(self):
        class _CompositeId(EntityId):
            tenant: str
            resource: str

        cid = _CompositeId(tenant="acme", resource="item-1")
        assert cid._key() == "acme#item-1"

    def test_none_field_raises(self):
        class _NullableId(EntityId):
            value: str | None = None

        with pytest.raises(EntityId.KeyDefinitionError):
            _NullableId()._key()


# ---------------------------------------------------------------------------
# RootEntity
# ---------------------------------------------------------------------------


class TestRootEntity:
    def test_default_version_is_zero(self):
        entity = _SampleEntity(id=_SampleId(value="e1"), name="test")
        assert entity.version == 0

    def test_increase_version(self):
        entity = _SampleEntity(id=_SampleId(value="e1"), name="test")
        entity._increase_version()
        assert entity.version == 1

    def test_created_and_last_update_set(self):
        entity = _SampleEntity(id=_SampleId(value="e1"), name="test")
        assert isinstance(entity.created, EpochTime)
        assert isinstance(entity.last_update, EpochTime)


# ---------------------------------------------------------------------------
# DomainAggregate
# ---------------------------------------------------------------------------


class TestDomainAggregate:
    def test_add_and_pull_events(self):
        agg = _SampleAggregate(id=_SampleId(value="a1"), name="agg")
        agg.add_event(_SampleEvent(payload="x"))
        assert len(agg.events) == 1

        pulled = agg.pull_events()
        assert len(pulled) == 1
        assert len(agg.events) == 0

    def test_pull_returns_copy_and_clears(self):
        agg = _SampleAggregate(id=_SampleId(value="a2"), name="agg")
        agg.add_event(_SampleEvent(payload="first"))
        pulled = agg.pull_events()
        agg.add_event(_SampleEvent(payload="second"))
        assert len(pulled) == 1
        assert pulled[0].payload == "first"
        assert agg.events[0].payload == "second"

    def test_events_start_empty_per_instance(self):
        a1 = _SampleAggregate(id=_SampleId(value="a1"), name="a1")
        a2 = _SampleAggregate(id=_SampleId(value="a2"), name="a2")
        a1.add_event(_SampleEvent(payload="only-for-a1"))
        assert len(a2.events) == 0


# ---------------------------------------------------------------------------
# DomainEvent
# ---------------------------------------------------------------------------


class TestDomainEvent:
    def test_auto_id_unique_per_instance(self):
        e1 = _SampleEvent(payload="a")
        e2 = _SampleEvent(payload="b")
        assert e1.id != e2.id

    def test_created_is_set(self):
        event = _SampleEvent(payload="x")
        assert isinstance(event.created, EpochTime)

    def test_is_frozen(self):
        event = _SampleEvent(payload="x")
        with pytest.raises(Exception):
            event.payload = "y"  # type: ignore


# ---------------------------------------------------------------------------
# split_list
# ---------------------------------------------------------------------------


class TestSplitList:
    def test_even_split(self):
        assert list(split_list([1, 2, 3, 4], chunk_size=2)) == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        assert list(split_list([1, 2, 3, 4, 5], chunk_size=2)) == [
            [1, 2], [3, 4], [5]]

    def test_empty_list(self):
        assert list(split_list([], chunk_size=3)) == []

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError):
            list(split_list([1], chunk_size=0))

    def test_chunk_size_larger_than_list(self):
        assert list(split_list([1, 2], chunk_size=10)) == [[1, 2]]


# ---------------------------------------------------------------------------
# update_last_update_date decorator
# ---------------------------------------------------------------------------


class TestUpdateLastUpdateDate:
    def test_updates_last_update_field(self):
        class _ExtendedEntity(_SampleEntity):
            @update_last_update_date
            def do_action(self) -> str:
                return "done"

        entity = _ExtendedEntity(id=_SampleId(value="x"), name="test")
        original_ts = entity.last_update.time_ns
        time.sleep(0.001)
        result = entity.do_action()

        assert result == "done"
        assert entity.last_update.time_ns > original_ts
