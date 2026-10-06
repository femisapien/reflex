"""HTTP adapter for Reflex with one explicit, disclosed capacity marker.

The Decision Index kit snapshot is unchanged. Its Reflex adapter predates the
wide-choice implementation's exact over-capacity wording, so we extend only
that terminal 422 classification. Other HTTP failures remain retryable errors.
"""

from __future__ import annotations

from decision_index.engines.base import Unsupported
from decision_index.engines.reflex_http import (
    REFLEX_CAPACITY_MARKERS,
    ReflexHttpSystemOne as FrozenReflexHttpSystemOne,
)


CAPACITY_MARKERS = REFLEX_CAPACITY_MARKERS + ("this tokenizer supports at most",)


class ReflexHttpSystemOne(FrozenReflexHttpSystemOne):
    def __call__(self, state, questions):
        response = self.client.post(
            "/v1/systemone",
            json={
                "model": self.model,
                "state": state,
                "questions": questions,
                **self.extra,
            },
        )
        if response.status_code in (400, 413, 422):
            message = response.text
            if any(marker in message for marker in CAPACITY_MARKERS):
                raise Unsupported(message)
        response.raise_for_status()
        raw = response.json()
        return {key: value for key, value in raw.items() if key != "evaluation_trace"}, raw
