import pytest
from aws_lambda_powertools import Logger, Metrics, Tracer

from erns_shared.aws.powertools import build_powertools


@pytest.fixture(autouse=True)
def powertools_env(monkeypatch):
    monkeypatch.setenv("POWERTOOLS_DEV", "true")
    monkeypatch.delenv("POWERTOOLS_SERVICE_NAME", raising=False)
    monkeypatch.delenv("POWERTOOLS_METRICS_NAMESPACE", raising=False)


class TestBuildPowertools:
    def test_returns_logger_tracer_metrics(self):
        logger, tracer, metrics = build_powertools(service="svc", namespace="NS")
        assert isinstance(logger, Logger)
        assert isinstance(tracer, Tracer)
        assert isinstance(metrics, Metrics)

    def test_logger_service_matches_argument(self):
        logger, _, _ = build_powertools(service="my-service", namespace="NS")
        assert logger.service == "my-service"

    def test_service_from_env_var(self, monkeypatch):
        monkeypatch.setenv("POWERTOOLS_SERVICE_NAME", "env-service")
        logger, _, _ = build_powertools(namespace="NS")
        assert logger.service == "env-service"

    def test_namespace_defaults_to_service_when_omitted(self, monkeypatch):
        monkeypatch.delenv("POWERTOOLS_METRICS_NAMESPACE", raising=False)
        _, _, metrics = build_powertools(service="fallback-svc")
        assert metrics.namespace == "fallback-svc"

    def test_namespace_from_env_var(self, monkeypatch):
        monkeypatch.setenv("POWERTOOLS_METRICS_NAMESPACE", "env-ns")
        _, _, metrics = build_powertools(service="svc")
        assert metrics.namespace == "env-ns"

    def test_explicit_args_override_env_vars(self, monkeypatch):
        monkeypatch.setenv("POWERTOOLS_SERVICE_NAME", "env-service")
        monkeypatch.setenv("POWERTOOLS_METRICS_NAMESPACE", "env-ns")
        logger, _, metrics = build_powertools(
            service="explicit-svc", namespace="explicit-ns"
        )
        assert logger.service == "explicit-svc"
        assert metrics.namespace == "explicit-ns"
