"""Multi-source confidence estimator for black-box LLM APIs.

Research basis:
- Verbalized ECE 0.377+ (arXiv 2412.14737, KDD 2025): verbalized confidence is a
  weak signal. Calibration bias of 0.25 corrects known over-confidence.
- P(True) (Kadavath et al. 2022): single follow-up call asking the model to
  self-evaluate; outperforms raw verbalized confidence.
- Adaptive semantic entropy (arXiv 2504.03579): start with 2 samples; stop early
  if they agree (Jaccard >= threshold), saving ~47% of sampling cost.
"""

from __future__ import annotations

import json
import re
from typing import Any

_UNCERTAINTY_MARKERS = [
    "i think",
    "i believe",
    "i'm not sure",
    "not certain",
    "approximately",
    "roughly",
    "unclear",
    "uncertain",
    "possibly",
    "probably",
    "might be",
    "could be",
    "unsure",
    "it appears",
    "it seems",
    "not confident",
]

P_TRUE_PROMPT = (
    "Review your previous answer to the question. "
    "How likely is it that your answer is factually correct and well-grounded? "
    'Reply with only a JSON object: {"p_true": <float 0.0 to 1.0>, "reason": "<one sentence>"}'
)


class UncertaintyEstimator:
    """Fuses verbalized confidence, P(True), and semantic entropy into one score.

    Args:
        enable_p_true: Whether to run the P(True) follow-up call.
        verbalized_weight: Weight for verbalized signal in fusion (default 0.4).
        p_true_weight: Weight for P(True) signal in fusion (default 0.6).
        verbalized_calibration_bias: Correction factor for verbalized ECE (0.25).
        p_true_calibration_bias: Correction factor for P(True) ECE (0.10).
        enable_semantic_sampling: Whether semantic entropy sampling is active.
        min_samples: Minimum samples before early-stop check.
        max_samples: Maximum samples to draw on disagreement.
        semantic_trigger_low: Lower bound of zone where sampling is triggered.
        semantic_trigger_high: Upper bound of zone where sampling is triggered.
        early_stop_jaccard: First-pair agreement threshold for early stop.
    """

    def __init__(
        self,
        *,
        enable_p_true: bool = True,
        verbalized_weight: float = 0.4,
        p_true_weight: float = 0.6,
        verbalized_calibration_bias: float = 0.25,
        p_true_calibration_bias: float = 0.10,
        enable_semantic_sampling: bool = False,
        min_samples: int = 2,
        max_samples: int = 5,
        semantic_trigger_low: float = 0.50,
        semantic_trigger_high: float = 0.72,
        early_stop_jaccard: float = 0.60,
    ) -> None:
        self.enable_p_true = enable_p_true
        self.verbalized_weight = verbalized_weight
        self.p_true_weight = p_true_weight
        self.verbalized_calibration_bias = verbalized_calibration_bias
        self.p_true_calibration_bias = p_true_calibration_bias
        self.enable_semantic_sampling = enable_semantic_sampling
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.semantic_trigger_low = semantic_trigger_low
        self.semantic_trigger_high = semantic_trigger_high
        self.early_stop_jaccard = early_stop_jaccard

    def extract_verbalized(self, text: str) -> float:
        """Parse raw confidence from LLM response text.

        Tries JSON parse, regex JSON search, heuristic marker counting, fallback.

        Returns:
            Raw (uncalibrated) float in [0.0, 1.0].
        """
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                val = obj.get("confidence") or obj.get("confidence_score")
                if val is not None:
                    return float(max(0.0, min(1.0, val)))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        match = re.search(r'\{[^}]*"confidence(?:_score)?"\s*:\s*([0-9.]+)[^}]*\}', text)
        if match:
            try:
                return float(max(0.0, min(1.0, float(match.group(1)))))
            except ValueError:
                pass

        lower = text.lower()
        hits = sum(1 for marker in _UNCERTAINTY_MARKERS if marker in lower)
        if hits > 0:
            return float(max(0.2, 0.7 - hits * 0.05))

        return 0.5

    async def p_true(
        self,
        original_messages: list[Any],
        response_text: str,
        model_client: Any,
        model_id: str,
    ) -> float:
        """Make one follow-up call asking the model to self-evaluate.

        Returns:
            Raw (uncalibrated) P(True) float; 0.5 on any failure.
        """
        if model_client is None:
            return 0.5
        try:
            from nexus.core.types import Message, Role

            messages = list(original_messages) + [
                Message(role=Role.ASSISTANT, content=response_text),
                Message(role=Role.USER, content=P_TRUE_PROMPT),
            ]
            resp = await model_client.complete(messages=messages, model=model_id)
            obj = json.loads(resp.content or "")
            val = float(obj["p_true"])
            return float(max(0.0, min(1.0, val)))
        except Exception:  # noqa: BLE001
            return 0.5

    def fuse(
        self,
        verbalized_raw: float,
        p_true_raw: float,
        *,
        p_true_ran: bool,
    ) -> float:
        """Weighted, calibrated fusion of verbalized and P(True) scores.

        Returns:
            Fused confidence float in [0.0, 1.0].
        """
        cal_v = self._calibrate(verbalized_raw, self.verbalized_calibration_bias)
        if not p_true_ran:
            return cal_v
        cal_p = self._calibrate(p_true_raw, self.p_true_calibration_bias)
        return self.verbalized_weight * cal_v + self.p_true_weight * cal_p

    async def semantic_entropy(
        self,
        messages: list[Any],
        model_client: Any,
        model_id: str,
    ) -> tuple[float, int]:
        """Adaptive semantic entropy estimation.

        Samples min_samples responses; if first pair agrees (Jaccard >= threshold),
        stops early. Otherwise extends to max_samples.

        Returns:
            (confidence, samples_used) where confidence = 1.0 - H_norm.
        """
        import math

        responses: list[str] = []
        for _ in range(self.min_samples):
            text = await self._sample_once(messages, model_client, model_id)
            responses.append(text)

        if len(responses) >= 2:
            j = _jaccard(responses[0], responses[1])
            if j >= self.early_stop_jaccard:
                return self._calibrate(1.0, 0.0), len(responses)

        while len(responses) < self.max_samples:
            text = await self._sample_once(messages, model_client, model_id)
            responses.append(text)

        clusters = _cluster_responses(responses)
        n = len(responses)
        h = 0.0
        for members in clusters:
            p_i = len(members) / n
            if p_i > 0:
                h -= p_i * math.log2(p_i)

        h_norm = h / math.log2(n) if n > 1 else 0.0
        confidence = self._calibrate(1.0 - h_norm, 0.0)
        return confidence, len(responses)

    def should_sample(self, fused_confidence: float) -> bool:
        """True when fused_confidence is in the semantic trigger zone."""
        return self.semantic_trigger_low <= fused_confidence <= self.semantic_trigger_high

    def _calibrate(self, raw: float, bias: float) -> float:
        """Apply ECE bias correction: calibrated = raw * (1.0 - bias)."""
        return float(max(0.0, min(1.0, raw * (1.0 - bias))))

    async def _sample_once(
        self,
        messages: list[Any],
        model_client: Any,
        model_id: str,
    ) -> str:
        """Call model_client once at temperature=0.8 and return response text."""
        try:
            resp = await model_client.complete(
                messages=messages,
                model=model_id,
                **{"temperature": 0.8},
            )
        except TypeError:
            resp = await model_client.complete(messages=messages, model=model_id)
        return resp.content or ""


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between word sets of two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa and not wb:
        return 1.0
    union = wa | wb
    if not union:
        return 0.0
    return len(wa & wb) / len(union)


def _cluster_responses(responses: list[str]) -> list[list[str]]:
    """Greedy cluster: assign to first cluster with Jaccard >= 0.4, else new cluster."""
    clusters: list[list[str]] = []
    for resp in responses:
        placed = False
        for cluster in clusters:
            if _jaccard(resp, cluster[0]) >= 0.4:
                cluster.append(resp)
                placed = True
                break
        if not placed:
            clusters.append([resp])
    return clusters
