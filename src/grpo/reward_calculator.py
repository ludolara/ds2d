from src.grpo.grpo_evaluator import GRPOEvaluator
from typing import Any, Callable, Dict, List

class RewardCalculator:
    def __init__(self):
        self._cache: Dict[str, Any] = {"stats": None, "key": None}

    def _compute_stats(self, completions: List[Any], **kwargs: Any) -> List[Dict[str, Any]]:
        key = id(completions[0] + completions[-1])
        if self._cache["key"] != key:
            self._cache["stats"] = [
                GRPOEvaluator.evaluate(
                    comp,
                    {
                        "room_count": rc,
                        "total_area": ta,
                        "input_graph": ig,
                        "rooms": rooms
                    }
                )
                for comp, rc, ta, ig, rooms in zip(
                    completions,
                    kwargs.get("room_count", []),
                    kwargs.get("total_area", []),
                    kwargs.get("input_graph", {}),
                    kwargs.get("rooms", [])
                )
            ]
            self._cache["key"] = key
        return self._cache["stats"]

    def _linear_reward(self, value: float, target: float = 1.0, round_digits: int = 4) -> float:
        diff = abs(value - target)
        reward = 1.0 if diff == 0.0 else max(0.0, 1.0 - diff)
        return round(reward, round_digits)

    def _valid_or_zero(self, stat: Dict[str, Any], fn: Callable[[Dict[str, Any]], float]) -> float:
        return fn(stat) if stat.get("is_valid_json", False) else 0.0

    def json_validity(self, completions: List[Any], **kwargs: Any) -> List[float]:
        stats = self._compute_stats(completions, **kwargs)
        return [1.0 if s.get("is_valid_json", False) else 0.0 for s in stats]

    def room_count(self, completions: List[Any], **kwargs: Any) -> List[float]:
        stats = self._compute_stats(completions, **kwargs)
        rewards = [
            self._valid_or_zero(
                s,
                lambda st: 1.0 if st.get("room_count", False) else 0.0
            )
            for s in stats
        ]
        return rewards

    def total_area(self, completions: List[Any], **kwargs: Any) -> List[float]:
        stats = self._compute_stats(completions, **kwargs)
        rewards = [
            self._valid_or_zero(
                s,
                lambda st: self._linear_reward(st.get("total_area", 0.0))
            )
            for s in stats
        ]
        return rewards

    def is_overlap(self, completions: List[Any], **kwargs: Any) -> List[float]:
        stats = self._compute_stats(completions, **kwargs)
        rewards = [
            self._valid_or_zero(
                s,
                lambda st: 1.0 if not st.get("is_overlap", True) else 0.0
            )
            for s in stats
        ]
        return rewards

    def compatibility(self, completions: List[Any], **kwargs: Any) -> List[float]:
        stats = self._compute_stats(completions, **kwargs)
        rewards = [
            self._valid_or_zero(
                s,
                lambda st: st.get("compatibility", 0.0)
            )
            for s in stats
        ]
        return rewards

    def make_reward_funcs(self) -> List[Callable[..., List[float]]]:
        return [
            self.json_validity,
            self.room_count,
            self.total_area,
            self.is_overlap,
            self.compatibility
        ]
