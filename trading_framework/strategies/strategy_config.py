"""Strategy configuration model.

This module defines the StrategyConfig schema used to parse and normalize
strategy-related configuration from JSON into engine-consumable parameters.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrategyConfig(BaseModel):
    """Strategy config that collects arbitrary extra keys into ``params``.

    JSON example:
        "strategy": {
          "class_path": "my_strategies.debug:DebugStrategy",
          "spread": 50.0,
          "size": 0.0001
        }

    Result:
        class_path="my_strategies.debug:DebugStrategy"
        params={"spread": 50.0, "size": 0.0001}
    """

    class_path: str = Field(..., min_length=1)

    params: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _collect_extras_into_params(cls, data: Any) -> Any:
        """Collect unknown top-level keys into the ``params`` mapping.

        This allows flat JSON strategy configuration without requiring
        a nested "params" object.
        """
        if not isinstance(data, dict):
            return data

        d = dict(data)

        explicit_params = d.pop("params", None)

        reserved = {"class_path"}

        extras = {k: v for k, v in d.items() if k not in reserved}

        for k in extras.keys():
            d.pop(k, None)

        merged: dict[str, Any] = {}
        if isinstance(explicit_params, dict):
            merged.update(explicit_params)
        merged.update(extras)

        d["params"] = merged
        return d

    def to_engine_params(self) -> dict[str, Any]:
        """Return a shallow copy of strategy parameters.

        The engine must not mutate configuration state.
        """
        return dict(self.params)
