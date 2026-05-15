from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


NASDAQ_ALL_TOKEN = "NASDAQ_ALL"

DEFAULT_TICKERS = [NASDAQ_ALL_TOKEN]


@dataclass
class DataConfig:
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    max_tickers: int = 40
    benchmark: str = "SPY"
    start: str = "2020-01-01"
    end: str = "2025-12-31"
    interval: str = "1d"
    auto_adjust: bool = False


@dataclass
class FactorConfig:
    # Local self-contained snapshot of the audited 526-factor registry.
    alpha526_path: str = "usalpha/alpha526_specs.json"


@dataclass
class TrainConfig:
    label_horizon: int = 1
    train_start: str = "2020-01-01"
    train_end: str = "2023-12-31"
    top_quantile: float = 0.2
    random_state: int = 42
    max_iter: int = 200
    learning_rate: float = 0.05
    max_depth: int = 6


@dataclass
class IOConfig:
    output_dir: str = "./artifacts"
    save_predictions: bool = True
    save_factor_snapshot: bool = False


@dataclass
class USAlphaConfig:
    data: DataConfig = field(default_factory=DataConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    io: IOConfig = field(default_factory=IOConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "USAlphaConfig":
        data = DataConfig(**payload.get("data", {}))
        factor = FactorConfig(**payload.get("factor", {}))
        train = TrainConfig(**payload.get("train", {}))
        io = IOConfig(**payload.get("io", {}))
        return cls(data=data, factor=factor, train=train, io=io)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_alpha526_path(self, project_root: Path) -> Path:
        path = Path(self.factor.alpha526_path).expanduser()
        if not path.is_absolute():
            path = (project_root / path).resolve()
        return path
