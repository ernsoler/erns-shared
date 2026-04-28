import os
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.typing import LambdaContext

__all__ = [
    "Logger",
    "Tracer",
    "Metrics",
    "MetricUnit",
    "LambdaContext",
    "build_powertools",
]


def build_powertools(
    service: str | None = None,
    namespace: str | None = None,
) -> tuple[Logger, Tracer, Metrics]:
    """Return a pre-configured (Logger, Tracer, Metrics) triple for a Lambda function.

    Falls back to POWERTOOLS_SERVICE_NAME / POWERTOOLS_METRICS_NAMESPACE env vars
    when arguments are omitted — so individual Lambdas don't need to repeat config.
    """
    svc = service or os.environ.get("POWERTOOLS_SERVICE_NAME", "service")
    ns = namespace or os.environ.get("POWERTOOLS_METRICS_NAMESPACE", svc)
    return Logger(service=svc), Tracer(service=svc), Metrics(namespace=ns, service=svc)
