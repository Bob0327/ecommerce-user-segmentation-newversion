# 电商用户价值分层与精准营销（最小完整版）

## 业务问题
营销预算有限，发券应该瞄准"消费金额最高的人"，还是"被券撬动空间最大的人"？

## 核心结论（模拟数据，4000 用户）
- 传统做法（RFM 前 20% 无差别发券）：ROI **-57%**，越有钱越不该发。
- 数据驱动六段式（只对 158 人高值流失群发券）：ROI **+188%**。
- 关键机制：发券利润 = 增量转化 × 毛利 − 券成本，决策变量是**增量（uplift）**而非复购概率——按复购概率排序的四种策略利润全部为负。

## 最小逻辑链（四条主干，缺一不可）
1. **SQL 取数**（`sql/pipeline_01_wide_table.sql`）：四张明细表 → 一张用户宽表（R/F/M/I/L + 实验分组），SQL 与 Python 的交接面。
2. **分层 + 建模**（`tools/pipeline_modeling.py` 第 2-4 步）：RFM+I 规则六段；KMeans 仅作对照（ARI=0.34，证明规则分层有结构但不是唯一解）；逻辑回归验证 I/L/人货匹配对复购的预测力（CV AUC 0.670→0.697）。
3. **随机实验 + ROI**（第 5 步）：实验表随机分流 → 分段增量 → 净利润/ROI；附方法诚实度（CI、功效）。
4. **看板**（第 6-7 步）：看板数据 CSV → Plotly 单文件 HTML（13 图 + 2 表）。

## 运行
```
pip install -r requirements.txt
python tools/pipeline_modeling.py    # 或双击 run_pipeline.bat（跑完自动打开看板）
```
端到端约 15 秒：建宽表 → 分层 → 建模 → 实验 ROI → 导出看板。

## 项目结构
```
user_value_segmentation/
├── sql/pipeline_01_wide_table.sql   # 唯一 SQL：宽表取数（含 @section 分区）
├── tools/pipeline_modeling.py       # 唯一脚本：7 步流水线
├── data/demo/ecommerce.db           # SQLite 模拟数据（users/transactions/sessions/experiment 四表）
├── data/dashboard/*.csv             # 第 6 步产物：看板数据
├── reports/dashboard.html           # 第 7 步产物：单文件看板（双击即开）
├── index.html                       # GitHub Pages 首页跳转
└── run_pipeline.bat                 # Windows 一键运行
```

## 口径声明
- **模拟数据只证明流程正确，不证明规律可推广**；158 人子组的增量 CI 为 [0.32, 31.35]pp、统计功效仅 0.56，属探索性结论，接真实数据须重跑并优先补对照实验。
- 商业假设（毛利率 0.35、券成本 12 元等）集中在 `pipeline_modeling.py` 顶部常量区，敏感性分析已验证结论在合理区间内稳健。
