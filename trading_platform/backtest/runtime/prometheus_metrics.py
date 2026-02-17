from __future__ import annotations

import json
import logging
import os

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

LOGGER = logging.getLogger(__name__)


class PrometheusMetricsClient:
    """Minimal Prometheus Pushgateway client for batch-style jobs.

    Expected environment:
    - PROMETHEUS_PUSHGATEWAY_URL: URL to the Pushgateway.
      Example: http://pushgateway.monitoring.svc.cluster.local:9091

    Optional:
    - PROMETHEUS_PUSHGATEWAY_GROUPING_KEY_JSON: JSON object used as grouping key.
      If not set, metrics are grouped only by the 'job' argument, which often
      causes pushes from different pods to overwrite each other.

      Example:
        {"workflow_uid": "ARGO_WORKFLOW_UID"}

    This client is intentionally best-effort: callers should treat it as a
    side-effect and never fail the workflow because of metrics delivery.
    """

    def __init__(self) -> None:
        self._pushgateway_url = os.environ.get("PROMETHEUS_PUSHGATEWAY_URL")
        self._grouping_key = self._load_grouping_key()
        self._registry = CollectorRegistry()

    def is_enabled(self) -> bool:
        return self._pushgateway_url is not None

    @staticmethod
    def _load_grouping_key() -> dict[str, str]:
        raw = os.environ.get("PROMETHEUS_PUSHGATEWAY_GROUPING_KEY_JSON")
        if not raw:
            return {}

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning(
                "Invalid PROMETHEUS_PUSHGATEWAY_GROUPING_KEY_JSON; ignoring"
            )
            return {}

        if not isinstance(data, dict):
            return {}

        grouping: dict[str, str] = {}
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, str):
                grouping[key] = value
        return grouping

    def push_gauge(
        self,
        *,
        name: str,
        value: float,
        labels: dict[str, str],
    ) -> None:
        if not self._pushgateway_url:
            return

        gauge = Gauge(
            name,
            documentation=name,
            labelnames=list(labels.keys()),
            registry=self._registry,
        )

        gauge.labels(**labels).set(value)

    def push_all(self, *, job: str) -> None:
        if not self._pushgateway_url:
            return

        push_to_gateway(
            gateway=self._pushgateway_url,
            job=job,
            registry=self._registry,
            grouping_key=self._grouping_key,
        )

        LOGGER.info(
            "Prometheus metrics pushed",
            extra={"job": job, "grouping_key": self._grouping_key},
        )
