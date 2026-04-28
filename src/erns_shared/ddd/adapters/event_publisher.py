import pydantic
import logging
import json
from typing import Protocol, List, TypeVar, Any, Dict
import erns_shared.ddd.base_types as base_types
import boto3
import backoff

_LOGGER = logging.getLogger("event_publisher")
E = TypeVar("E", bound=base_types.DomainEvent)


class EventPublishError(Exception):
    ...


class EventPublisher(Protocol):
    def publish(self, events: List[E]) -> None:
        ...


class _CommonSettings(base_types.Settings):
    backoff_default_tries: int = 3
    backoff_default_max_time: int = 3


_COMMON_SETTINGS = _CommonSettings()


class EventBridgePublisher:
    class _Settings(base_types.Settings):
        event_bridge_topic_arn: str = "TO-FILL"

    class EventBody(base_types.ValueObject):
        eventbus_name_arn: str = pydantic.Field(alias="EventBusName")
        source: str = pydantic.Field(alias="Source")
        detail_type: str = pydantic.Field(alias="DetailType")
        detail: str = pydantic.Field(alias="Detail")

    def __init__(self) -> None:
        self._settings = EventBridgePublisher._Settings()
        self._client = boto3.client("events")

    @backoff.on_exception(
        backoff.fibo,
        EventPublishError,
        max_tries=_COMMON_SETTINGS.backoff_default_tries,
        max_time=_COMMON_SETTINGS.backoff_default_max_time,
    )
    def publish(self, events: List[E]) -> None:
        _LOGGER.info(f"EventBridge publisher. Events qty [{len(events)}]")
        if not events:
            _LOGGER.warning("No events provided. List passed is empty")
            return

        try:
            events_body_parsed = [
                self._to_eventbridge_entry(domain_event=event) for event in events
            ]
            response = self._put_events(events=events_body_parsed)
            _LOGGER.info(f"EVENT BRIDGE RESPONSE: {str(response)}")
            if response["FailedEntryCount"] == 0:
                _LOGGER.info("Events published successfully")
            else:
                for entry in response["Entries"]:
                    if "ErrorCode" in entry:
                        _LOGGER.error(
                            f"Failed to publish event: {entry['ErrorCode']} - {entry['ErrorMessage']}"
                        )
                raise EventPublishError()
        except EventPublishError:
            raise
        except Exception as ex:
            _LOGGER.error("Error when trying to publish domain event to EventBridge")
            raise EventPublishError() from ex

    def _to_eventbridge_entry(self, domain_event: E) -> Dict[str, Any]:
        return EventBridgePublisher.EventBody(
            EventBusName=self._settings.event_bridge_topic_arn,
            Source=domain_event.domain_name,
            DetailType=type(domain_event).__name__,
            Detail=json.dumps(domain_event.model_dump()),
        ).model_dump(by_alias=True)

    def _put_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._client.put_events(Entries=events)  # type: ignore
