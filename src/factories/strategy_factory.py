from src.active_learning.strategies.coreset_sampling import CoreSetSampling
from src.active_learning.strategies.entropy_sampling import EntropySampling
from src.active_learning.strategies.margin_sampling import MarginSampling
from src.active_learning.strategies.random_sampling import RandomSampling


def create_strategy(name: str, **kwargs):
    strategy_name = name.lower()
    if strategy_name == "random":
        return RandomSampling(**kwargs)
    if strategy_name in {"margin", "uncertainty"}:
        return MarginSampling()
    if strategy_name == "entropy":
        return EntropySampling()
    if strategy_name in {"coreset", "core_set"}:
        return CoreSetSampling()
    raise ValueError(f"Unknown active-learning strategy: {name}")
