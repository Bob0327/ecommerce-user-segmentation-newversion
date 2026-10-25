# 电商用户价值分层与精准营销

SQL 取数 + Python 建模可视化的完整分析流水线：数据审计 → 特征宽表 → 规则分层与
KMeans 对照 → 逻辑回归 / uplift 建模 → A/B 增量验证 → ROI 与预算分配 → 敏感性分析。

## 快速开始

```bash
pip install -r requirements.txt
python tools/pipeline_modeling.py     # 一键跑完全流程
streamlit run dashboard/app.py        # 交互式看板（可选）
```

跑完自动生成 `reports/dashboard.html`（单文件交互看板，可传 GitHub Pages 供在线观看）。

## 目录结构

```
sql/pipeline_01_wide_table.sql   SQL 取数：审计 + 一人一行原始指标宽表
tools/pipeline_modeling.py       流水线主脚本：建模、分层、ROI、看板产出
dashboard/app.py                 Streamlit 交互看板
data/demo/                       演示数据（SQLite 库 + 4 张 CSV）
reports/dashboard.html            生成物：单文件看板
```

## 关键结果（演示数据）

| 指标 | 数值 |
|---|---|
| 发券整体增量 | +3.11 个百分点 |
| 高值流失群体增量 | +15.75 个百分点 |
| 六段式方案 ROI（1 万元预算） | 188% |
| 传统 RFM 前 20% 方案 ROI | -57% |
| 复购预测 AUC（逻辑回归 5 折 CV） | 0.697 |
| uplift 排序 Top 20% 发券净利润 | +1805 元 |

核心结论：应按"被券撬动的空间"（uplift）而非"消费价值"选择投放人群。

## 说明

- 数据为固定随机种子生成的模拟数据，仅用于演示分析流程，不代表真实业务结论
- 特征窗口 2025-01-01 ~ 2025-06-30，结果窗口 2025-07-01 ~ 2025-09-30
- 建模仅使用对照组，避免处理效应污染自然复购规律
