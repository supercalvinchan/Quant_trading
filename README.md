# USalpha

USalpha 是一个自包含的量化研究与回测工作台，覆盖数据加载、因子计算、模型训练、回测分析、可视化操作，以及 LLM 因子演进和 A 股页面化回测。

它适合作为研究原型、策略实验台和后续继续开发的框架底座。

## 功能

### 1. 研究主流程

- 加载行情数据
- 计算内置 `526` 因子
- 训练截面收益预测模型
- 输出回测结果与结构化产物

### 2. LLM 因子演进

- 生成候选因子表达式
- 校验表达式合法性与可计算性
- 按 IC / 回测表现打分
- 保存候选结果与入库因子

### 3. 可视化工作台

- 股票查看
- 因子训练与挖掘
- 中国股市回测子页面
- 候选股票、持仓明细、图表展示

### 4. A 股回测准备

- 本地 `parquet` 行情缓存
- 缓存优先 / 仅读缓存模式
- 适合大范围股票池的分步准备

### 5. 可迁移目录运行

- 入口脚本按仓库相对路径解析
- 改文件夹名或移动仓库后仍可运行

## 安装

```bash
cd /path/to/your/repo
python -m pip install -r requirements.txt
```

如需 A 股大规模缓存更新，建议确保运行环境安装了 `mootdx` 和 `akshare`。

## 快速开始

### 1. 检查迁目录后的可运行性

```bash
cd /path/to/your/repo
python scripts/check_portable_setup.py
```

若输出 `portable_setup=ok`，说明核心入口和相对路径解析正常。

### 2. 运行主研究流程

```bash
cd /path/to/your/repo
python scripts/run_repo.py pipeline -- \
  --tickers NASDAQ_ALL \
  --max-tickers 40 \
  --benchmark SPY \
  --start 2020-01-01 \
  --end 2025-12-31 \
  --train-end 2023-12-31
```

### 3. 运行 LLM 因子演进

```bash
cd /path/to/your/repo
GLM_API_KEY=your_key python scripts/run_repo.py llm-round -- \
  --tickers AAPL,MSFT,NVDA,AMZN,GOOGL \
  --start 2024-01-01 \
  --end 2026-04-15 \
  --num-candidates 72 \
  --top-k-accept 12
```

### 4. 启动可视化页面

```bash
cd /path/to/your/repo
python scripts/run_repo.py dashboard
```

启动后浏览器打开 `http://localhost:8501`。

## 使用方式

### 命令行

适合：
- 快速验证一轮训练 / 回测
- 批量留存 `artifacts`
- 接入其他脚本工作流

核心入口：
- `scripts/run_repo.py`
- `scripts/run_usalpha.py`
- `scripts/run_llm_factor_round.py`

### 页面操作

适合：
- 看股票
- 调整参数
- 跑中国股市回测
- 浏览训练与挖掘结果

核心页面：
- `apps/factor_dashboard.py`

### 继续开发

适合扩展：
- 数据层
- 因子层
- 模型层
- 回测层
- 可视化层
- 实验与协作规范

## A 股回测说明

- 大范围股票池建议先执行“仅建立/更新行情缓存”，再运行回测
- 全市场场景建议开启“仅读本地缓存”
- 日线结束日期会按最近已完成交易日处理，避免对不存在的当日 K 线反复刷新
- 如果缓存损坏或缺失过多，应优先补缓存，不要直接全市场联网回测

## 输出产物

### 主流程产物

每次运行会在 `artifacts/run_YYYYMMDD_HHMMSS/` 下生成：

- `summary.json`
- `factor_stats.json`
- `model_metrics.json`
- `backtest_metrics.json`
- `backtest_daily.csv`
- `predictions.csv`
- `run_config.json`

### LLM 因子演进产物

每次运行会在 `artifacts/evolution_round_*/` 下生成：

- `prompt.txt`
- `candidates_parsed.json`
- `candidates_scored.csv`
- `accepted_factors.json`
- `llm_raw_response.json`
- `llm_error.txt`（如有）

## 目录结构

```text
USalpha/
  apps/
    factor_dashboard.py
  scripts/
    run_repo.py
    run_usalpha.py
    run_llm_factor_round.py
    run_dashboard.sh
    check_portable_setup.py
  usalpha/
    config.py
    data.py
    factors.py
    model.py
    backtest.py
    pipeline.py
    factor_evolution.py
    dashboard_service.py
    alpha526_specs.json
    llm_factor_library.json
  artifacts/
  skills/
```

## 开发与协作

- 仓库内部资源默认按相对路径解析，不依赖固定绝对目录
- 推荐统一使用 `python scripts/run_repo.py ...` 启动内置入口
- 协作原则与检查清单位于 `skills/usalpha-dev-principles/`
- 分析结果应明确区分真实数据结果与 fallback 结果，不要混写

## 版本记录

### v1

当前版本包含以下功能与改进：

1. 研究主流程
   - 内置 526 因子计算
   - 截面收益预测模型
   - 回测与结构化产物输出

2. LLM 因子演进
   - 候选表达式生成
   - 合法性校验
   - 因子评分与入库

3. Dashboard
   - 股票查看
   - 因子训练与挖掘页面
   - 中国股市回测子页面
   - 候选股票、持仓与图表展示

4. A 股数据准备能力
   - 本地行情缓存
   - 缓存优先 / 仅读缓存回测路径
   - 全市场使用场景的基础支持

5. 运行稳定性改进
   - 去除对固定仓库路径的依赖
   - 新增统一入口脚本 `run_repo.py`
   - 新增迁目录自检脚本 `check_portable_setup.py`
   - 浏览器心跳自动关停默认关闭
   - 回测中移除 `scipy` 必需依赖，Spearman IC 改为本地实现

6. 开发协作基础
   - 根目录操作型 skill
   - `skills/usalpha-dev-principles/` 开发原则与检查清单

后续版本建议继续按 `v2 / v3 / ...` 追加新增能力、行为变更和兼容性说明。
