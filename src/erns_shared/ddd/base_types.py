import enum
import decimal
import pydantic
import uuid
import time
import functools
from typing import Dict, Any, Iterator, List, Callable


class NamedEnum(enum.StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


class UUIDGenerator:
    @staticmethod
    def uuid() -> str:
        return str(uuid.uuid4())


class Inmutable(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="ignore",
        json_encoders={
            decimal.Decimal: str,
        },
    )


class ValueObject(Inmutable): ...


class EpochTime(Inmutable):
    time_ns: int

    def __str__(self) -> str:
        return str(self.time_ns)

    @staticmethod
    def now() -> "EpochTime":
        return EpochTime(time_ns=time.time_ns())


def update_last_update_date(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.last_update = EpochTime.now()
        return result

    return wrapper


class Key(Inmutable):
    class KeyDefinitionError(Exception):
        def __init__(self) -> None:
            super().__init__(
                "Key only support primitives types. None value is not allowed"
            )

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        return super().model_dump(**kwargs)

    def _key(self, **kwargs: Any) -> str:
        attrs_dict = super().model_dump(**kwargs)

        def values(attrs_dict: Dict[str, Any]) -> Iterator[str]:
            for v in attrs_dict.values():
                if not v or not isinstance(
                    v, (str, int, float, decimal.Decimal, enum.Enum)
                ):
                    raise EntityId.KeyDefinitionError()
                yield str(v.value if isinstance(v, enum.Enum) else v)

        return "#".join(values(attrs_dict=attrs_dict))


class EntityId(Key): ...


class Entity(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra="ignore",
        json_encoders={
            decimal.Decimal: str,
        },
    )
    id: "EntityId"


class RootEntity(Entity):
    created: EpochTime = pydantic.Field(default_factory=EpochTime.now)
    last_update: EpochTime = pydantic.Field(default_factory=EpochTime.now)
    version: int = pydantic.Field(default=0)

    def _increase_version(self) -> None:
        self.version += 1

    def dict(self, **kwargs: Any) -> Dict[str, Any]:
        return super().model_dump(**kwargs)


class DomainAggregate(RootEntity):
    _events: List["DomainEvent"] = pydantic.PrivateAttr(default_factory=list)

    @property
    def events(self) -> List["DomainEvent"]:
        return self._events

    def add_event(self, event: "DomainEvent") -> None:
        self._events.append(event)

    def pull_events(self) -> List["DomainEvent"]:
        events = self._events.copy()
        self._events.clear()
        return events


class RepositoryAggregate(RootEntity): ...


class Projection(DomainAggregate): ...


class Command(Inmutable): ...


class DomainEvent(Inmutable):
    id: str = pydantic.Field(default_factory=UUIDGenerator.uuid)
    created: EpochTime = pydantic.Field(default_factory=EpochTime.now)
    domain_name: str


def split_list(input_list: List[Any], chunk_size: int) -> Iterator[List[Any]]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be bigger than zero")
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i : i + chunk_size]
