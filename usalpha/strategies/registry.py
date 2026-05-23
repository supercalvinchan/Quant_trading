from __future__ import annotations

from dataclasses import dataclass

from .base import BaseStrategy
from .alpha526_number_rank import Alpha526NumberRankStrategy
from .factor_rank import FactorRankStrategy
from .institutional_crowding import (
    InstitutionalCrowdingStrategy,
    InstitutionalGrowthStrategy,
    InstitutionalWhiteHorseStrategy,
)
from .small_cap_timing import SmallCapTimingStrategy
from .technical_score import TechnicalScoreStrategy


@dataclass(frozen=True)
class StrategySpec:
    type: str
    name: str
    description: str
    explanation: str
    strategy_cls: type[BaseStrategy]


STRATEGY_REGISTRY: dict[str, StrategySpec] = {
    "technical_score": StrategySpec(
        type="technical_score",
        name="技术指标综合分",
        description="基于 MACD/KDJ/RSI/DMI/WR/均线的综合评分做横截面排序。",
        explanation=(
            "## 数学信号定义\n"
            "- `EMA12_t = EWM(close, span=12)`，`EMA26_t = EWM(close, span=26)`\n"
            "- `MACD_t = EMA12_t - EMA26_t`\n"
            "- `SIGNAL_t = EWM(MACD, span=9)`\n"
            "- `HIST_t = MACD_t - SIGNAL_t`\n"
            "- `RSV_t = (close_t - rolling_min(low,9)) / (rolling_max(high,9)-rolling_min(low,9)) * 100`\n"
            "- `K_t = EWM(RSV, alpha=1/3)`，`D_t = EWM(K, alpha=1/3)`，`J_t = 3K_t - 2D_t`\n"
            "- `RSI_n = 100 - 100 / (1 + EWM(gain,1/n) / EWM(loss,1/n))`，这里使用 `n=6,12,24`\n"
            "- `PDI/MDI/ADX` 按标准 DMI 定义：\n"
            "  `TR = max(high-low, |high-close[-1]|, |low-close[-1]|)`，\n"
            "  `PDI = 100 * EWM(+DM,1/14) / ATR`，`MDI = 100 * EWM(-DM,1/14) / ATR`，\n"
            "  `ADX = EWM(|PDI-MDI|/(PDI+MDI)*100, 1/14)`\n"
            "- `WR_n = (rolling_max(high,n)-close) / (rolling_max(high,n)-rolling_min(low,n)) * 100`，这里用 `n=10,20`\n"
            "- 均线组：`MA5, MA10, MA20, MA120`\n\n"
            "## 分项打分\n"
            "- `MACD_score`:\n"
            "  - 若 `MACD > SIGNAL`，基准 `+35`，否则 `-35`\n"
            "  - 当日金叉额外 `+45`；死叉额外 `-45`\n"
            "  - 若 `HIST_t > HIST_{t-1}` 再加 `+20`，否则 `-20`\n"
            "- `KDJ_score`:\n"
            "  - 中心项：`(50-|K-50|)*0.4`\n"
            "  - 若 `K>D` 加 `+25`，否则 `-25`\n"
            "  - 若 `J<20` 加 `+35`；若 `J>80` 加 `-35`\n"
            "- `RSI_score`:\n"
            "  - 使用 `RSI6`\n"
            "  - 若 `RSI6<30` 记 `+70`；若 `RSI6>70` 记 `-70`；否则记 `(50-RSI6)*1.2`\n"
            "  - 若 `RSI6_t > RSI6_{t-1}` 再加 `+15`，否则 `-15`\n"
            "- `DMI_score`:\n"
            "  - 若 `PDI>MDI` 基准 `+30`，否则 `-30`\n"
            "  - 再叠加趋势强度项：若 `PDI>MDI` 加 `+clip(ADX,40)`，否则减去该值\n"
            "- `WR_score`:\n"
            "  - 使用 `WR10`\n"
            "  - 若 `WR>80` 记 `+70`；若 `WR<20` 记 `-70`；否则记 `(50-WR)*(-1.2)`\n"
            "- `MA_score`:\n"
            "  - `close>MA5` 加 `+25`，否则 `-25`\n"
            "  - `MA5>MA10` 加 `+25`，否则 `-25`\n"
            "  - `MA10>MA20` 加 `+25`，否则 `-25`\n"
            "  - `close>MA120` 加 `+25`，否则 `-25`\n\n"
            "## 最终分数\n"
            "- 先将每个分项分数裁剪到 `[-100,100]`\n"
            "- 最终 `Score_t = mean(MACD_score, KDJ_score, RSI_score, DMI_score, WR_score, MA_score)`\n"
            "- 每天对股票横截面按 `Score_t` 从高到低排序。分数越高，表示趋势、动量、均线结构与超买超卖状态越偏强。"
        ),
        strategy_cls=TechnicalScoreStrategy,
    ),
    "factor_rank": StrategySpec(
        type="factor_rank",
        name="单因子表达式排序",
        description="对预定 Qlib/base 风格因子表达式做逐日横截面排序。",
        explanation=(
            "## 数学信号定义\n"
            "- 该策略直接把你输入的表达式当作日频因子 `f_t(i)`，其中 `i` 是股票，`t` 是日期。\n"
            "- 基础字段：`$open, $high, $low, $close, $vwap, $volume`\n"
            "- 主要算子：\n"
            "  - `Ref(x,n) = x_{t-n}`\n"
            "  - `Mean(x,n) = rolling_mean(x,n)`\n"
            "  - `Std(x,n) = rolling_std(x,n)`\n"
            "  - `Max(x,n), Min(x,n)` 为滚动极值\n"
            "  - `Slope(x,n)` 为窗口内对时间序列做一元线性回归得到的斜率\n"
            "  - `Rsquare(x,n)` 为同一回归的 `R^2`\n"
            "  - `Resi(x,n)` 为窗口最后一点相对回归拟合值的残差\n\n"
            "## 典型例子\n"
            "- 价格动量：`$close / Ref($close, 20) - 1`\n"
            "- 均线偏离：`$close / Mean($close, 20) - 1`\n"
            "- 波动收缩：`-Std($close / Ref($close,1) - 1, 20)`\n"
            "- 趋势斜率：`Slope($close, 20)`\n\n"
            "## 最终分数\n"
            "- 对每只股票逐日计算表达式值，得到 `Score_t(i) = f_t(i)`\n"
            "- 再在每个交易日做横截面排序\n"
            "- 该策略本身不做标准化、截尾或行业中性化，最终排序完全由表达式数值大小决定。"
        ),
        strategy_cls=FactorRankStrategy,
    ),
    "alpha526_number_rank": StrategySpec(
        type="alpha526_number_rank",
        name="526因子编号回测",
        description="从本地 526 因子库中按编号选择一个因子，直接做横截面排序回测。",
        explanation=(
            "## 数学信号定义\n"
            "- 设你输入的编号为 `k`，系统从本地 `alpha526` 因子库读取第 `k` 个因子的定义 `f_k(i,t)`。\n"
            "- 对每只股票 `i`、每个交易日 `t`，直接计算该因子的原始值作为分数：\n"
            "  `Score_t(i) = f_k(i,t)`\n"
            "- 然后按当天的横截面分数从高到低排序。\n\n"
            "## 使用方式\n"
            "- 你只需要输入因子编号，例如 `1`、`128`、`526`\n"
            "- 页面会显示该编号对应的：`name / category / expression`\n"
            "- 这样你可以稳定复现实验，不用手工拷贝长公式。\n\n"
            "## 适用范围\n"
            "- `base` 因子：直接由 OHLCV / VWAP 等基础字段计算\n"
            "- `alpha101` 因子：按本地 alpha101 evaluator 计算\n"
            "- `alpha191` 因子：按本地 alpha191 evaluator 计算，依赖基准序列\n"
        ),
        strategy_cls=Alpha526NumberRankStrategy,
    ),
    "small_cap_timing": StrategySpec(
        type="small_cap_timing",
        name="小市值轮动",
        description="按总市值从小到大选股，可叠加成交额过滤，适合A股小市值轮动回测。",
        explanation=(
            "## 数学信号定义\n"
            "- 记历史总市值序列为 `MV_t(i)`，单位为亿元\n"
            "- 基础原始分数定义为：`Score_t(i) = - MV_t(i)`\n"
            "- 因为做横截面排序时分数越高越靠前，所以用负号把“小市值优先”转成“大分数优先”\n\n"
            "## 过滤条件\n"
            "- 市值范围过滤：\n"
            "  - `MV_t(i) >= min_total_value_yi`\n"
            "  - `MV_t(i) <= max_total_value_yi`\n"
            "- 流动性过滤（若开启）：\n"
            "  - `AvgAmount_t(i) = rolling_mean(amount, amount_window)`\n"
            "  - 要求 `AvgAmount_t(i) >= min_avg_amount`\n"
            "- 名称过滤：若股票名包含 `ST` 或 `退`，则直接剔除\n\n"
            "## 最终分数\n"
            "- 满足过滤条件时：`Score_t(i) = -MV_t(i)`\n"
            "- 不满足过滤条件时：`Score_t(i) = NaN`\n"
            "- 每天对横截面按 `Score_t` 排序，等价于“在合格股票里按市值从小到大排序”。\n\n"
            "## 含义\n"
            "- 该策略不是复杂复合因子，而是一个非常直接的小盘容量暴露\n"
            "- 本质上是在做 `size factor` 的反向暴露：越小盘，分数越高。"
        ),
        strategy_cls=SmallCapTimingStrategy,
    ),
    "institutional_crowding": StrategySpec(
        type="institutional_crowding",
        name="机构抱团代理",
        description="用大市值、高成交额、低换手、强趋势、低短波动等代理特征识别机构重仓抱团股。",
        explanation=(
            "## 数学信号定义\n"
            "对每只股票 `i` 在日期 `t` 定义：\n"
            "- `MV_t(i)`：总市值（亿元）\n"
            "- `AvgAmount_t(i) = rolling_mean(amount, turnover_window)`\n"
            "- `TurnoverRatio_t(i) = AvgAmount_t(i) / (MV_t(i) * 1e8)`\n"
            "- `Momentum_t(i) = close_t / close_{t-momentum_window} - 1`\n"
            "- `MA_t(i) = rolling_mean(close, trend_window)`\n"
            "- `TrendGap_t(i) = close_t / MA_t(i) - 1`\n"
            "- `Vol_t(i) = rolling_std(close/close[-1]-1, vol_window)`\n\n"
            "## 横截面分位数化\n"
            "每天分别对以下量做横截面百分位排名：\n"
            "- `CapRank_t = rank_pct(MV_t)`\n"
            "- `AmountRank_t = rank_pct(AvgAmount_t)`\n"
            "- `TurnoverRank_t = rank_pct(TurnoverRatio_t)`\n"
            "- `MomentumRank_t = rank_pct(Momentum_t)`\n"
            "- `TrendRank_t = rank_pct(TrendGap_t)`\n"
            "- `VolRank_t = rank_pct(Vol_t)`\n\n"
            "## 过滤条件\n"
            "- `MV_t >= min_total_value_yi`\n"
            "- `AvgAmount_t >= min_avg_amount`\n"
            "- 若设置 `max_turnover_ratio`，要求 `TurnoverRatio_t <= max_turnover_ratio`\n"
            "- 名称过滤：剔除 `ST` 和 `退`\n\n"
            "## 最终分数\n"
            "`Score_t = 0.22*CapRank + 0.18*AmountRank + 0.22*(1-TurnoverRank) + 0.20*MomentumRank + 0.10*TrendRank + 0.08*(1-VolRank)`\n"
            "- 分数越高，表示越接近“大资金容量足、流动性足、筹码稳定、趋势持续、短波动低”的机构抱团特征。"
        ),
        strategy_cls=InstitutionalCrowdingStrategy,
    ),
    "institutional_white_horse": StrategySpec(
        type="institutional_white_horse",
        name="白马机构抱团",
        description="偏大市值、低换手、低波动、稳趋势，适合识别核心白马型机构重仓股。",
        explanation=(
            "## 数学信号定义\n"
            "该策略与“机构抱团代理”使用同一组原始信号：\n"
            "- `MV_t, AvgAmount_t, TurnoverRatio_t, Momentum_t, TrendGap_t, Vol_t`\n"
            "- 区别在于参数阈值和加权系数更偏向“核心白马”\n\n"
            "## 过滤参数\n"
            "- `MV_t >= 500` 亿元\n"
            "- `AvgAmount_t >= 5e8`\n"
            "- `TurnoverRatio_t <= 0.05`\n\n"
            "## 最终分数\n"
            "`Score_t = 0.28*CapRank + 0.18*AmountRank + 0.24*(1-TurnoverRank) + 0.12*MomentumRank + 0.08*TrendRank + 0.10*(1-VolRank)`\n\n"
            "## 含义\n"
            "- 更高权重给 `CapRank` 和 `低换手`\n"
            "- 更低权重给激进动量\n"
            "- 所以它偏向“容量大、换手低、波动低、上涨更稳”的白马核心资产。"
        ),
        strategy_cls=InstitutionalWhiteHorseStrategy,
    ),
    "institutional_growth": StrategySpec(
        type="institutional_growth",
        name="成长机构抱团",
        description="偏中大市值、高景气趋势、较强动量，适合识别机构偏爱的成长抱团股。",
        explanation=(
            "## 数学信号定义\n"
            "仍使用同一组原始信号：\n"
            "- `MV_t, AvgAmount_t, TurnoverRatio_t, Momentum_t, TrendGap_t, Vol_t`\n\n"
            "## 过滤参数\n"
            "- `MV_t >= 120` 亿元\n"
            "- `AvgAmount_t >= 2e8`\n"
            "- `TurnoverRatio_t <= 0.12`\n\n"
            "## 最终分数\n"
            "`Score_t = 0.14*CapRank + 0.18*AmountRank + 0.16*(1-TurnoverRank) + 0.26*MomentumRank + 0.18*TrendRank + 0.08*(1-VolRank)`\n\n"
            "## 含义\n"
            "- 相比白马版，降低了对绝对大市值的偏好\n"
            "- 明显提高了 `MomentumRank` 与 `TrendRank` 的权重\n"
            "- 因此更偏向“中大市值、趋势强、动量高、机构持续加仓”的成长抱团股。"
        ),
        strategy_cls=InstitutionalGrowthStrategy,
    ),
}


def get_strategy(strategy_type: str) -> BaseStrategy:
    key = str(strategy_type or "").strip().lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(f"unknown strategy_type: {strategy_type}")
    return STRATEGY_REGISTRY[key].strategy_cls()


def list_strategy_metadata() -> list[dict[str, str]]:
    return [
        {
            "type": spec.type,
            "name": spec.name,
            "description": spec.description,
            "explanation": spec.explanation,
        }
        for spec in STRATEGY_REGISTRY.values()
    ]
