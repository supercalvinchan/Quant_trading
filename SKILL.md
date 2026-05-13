---
name: usalpha-ops
description: 在 USalpha 仓库中执行美股数据导入、526 因子计算、训练与回测，并校验输出产物。
---

# USalpha Operational Skill

## 触发条件

- 需要在 `USalpha` 中跑一轮美股因子研究。
- 需要确认 526 因子是否计算完整。
- 需要快速定位运行输出和关键指标。

## 标准流程

1. 进入目录并安装依赖。
2. 运行 `scripts/run_usalpha.py`。
3. 打开 `artifacts/run_*/summary.json` 检查 `factor_count` 和回测指标。
4. 需要明细时查看 `predictions.csv` 与 `backtest_daily.csv`。
5. 需要挖掘新因子时运行 `scripts/run_llm_factor_round.py` 并检查 `artifacts/evolution_round_*/accepted_factors.json`。
6. 需要可视化全流程时运行 `bash scripts/run_dashboard.sh`，页面一键执行训练与因子挖掘。
7. 在可视化页面侧边栏可分别配置训练日期、回测日期、预测日期（互相独立）。

## 命令模板

```bash
cd /path/to/your/repo
python -m pip install -r requirements.txt
PYTHONPATH=. python scripts/run_usalpha.py \
  --tickers AAPL,MSFT,NVDA,AMZN,META \
  --benchmark SPY \
  --start 2020-01-01 \
  --end 2025-12-31 \
  --train-end 2023-12-31

bash scripts/run_dashboard.sh
```

## 数据与接口约定

- 行情面板格式：`index=datetime`，`columns=MultiIndex(instrument, field)`。
- 必须字段：`open/high/low/close/volume`，其余 `amount/vwap/ret` 由程序补齐。
- 因子结果格式：`index=(datetime, instrument)`，`columns=526 factors`。
- 526 因子定义位于本仓库 `usalpha/alpha526_specs.json`，不依赖相邻 `alpha_mining` 仓库。

## 产物检查清单

- `factor_stats.json`：`factor_count == 526`。
- `model_metrics.json`：确认 `samples_train/samples_test` 非 0。
- `backtest_metrics.json`：检查 `days`、`sharpe`、`max_drawdown`。
- `summary.json`：作为对外汇总入口。

## 常见问题

- `No module named yfinance`：先执行依赖安装。
- 缺少 `qlib` 或相邻 `alpha_mining` 仓库：不阻塞，`factors.py` 使用本地 526 定义和本地 alpha101/191 evaluator。
- 某些股票下载失败：会告警并自动剔除失败标的继续运行。
- Yahoo/GLM 返回 429：演进脚本会自动降级到 `local_fallback` 生成模式；配额恢复后可直接重跑同一命令。
