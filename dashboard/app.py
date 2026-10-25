# -*- coding: utf-8 -*-
"""
电商用户价值分层与精准营销 —— Streamlit 交互式看板
====================================================
用途：学习与面试展示。每个页签回答一个分析问题，并配"怎么读这张图"。

启动方式（在项目根目录执行）：
    streamlit run dashboard/app.py

数据来自 data/dashboard/*.csv，由 tools/build_dashboard_data.py 生成。
如果改了 SQL 或数据库，先重跑 build_dashboard_data.py，再刷新本页面。
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'dashboard'

st.set_page_config(
    page_title='电商用户价值分层看板',
    page_icon='📊',
    layout='wide',
)

SEGMENT_ORDER = ['核心价值', '活跃成长', '新近低值', '高值流失', '沉睡中值', '低值流失']
SEG_COLORS = {
    '核心价值': '#0B5CAD', '活跃成长': '#2E8B57', '新近低值': '#7FB3D5',
    '高值流失': '#E67E22', '沉睡中值': '#9B59B6', '低值流失': '#BDC3C7',
}


@st.cache_data
def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, encoding='utf-8-sig')


# ---------------------------------------------------------------------------
# 页头
# ---------------------------------------------------------------------------
st.title('📊 电商用户价值分层与精准营销 —— 分析看板')
st.caption('数据：模拟演示数据（4000 用户 / 7628 笔交易）｜ 特征窗口 2025-01-01~06-30，结果窗口 2025-07-01~09-30')

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    '0 项目概览', '1 数据审计', '2 用户分层', '3 A/B 实验', '4 ROI 与预算', '5 敏感性分析', '6 结论与提醒',
])

# ===========================================================================
# 页签 0：项目概览
# ===========================================================================
with tab0:
    st.subheader('这个项目的业务问题是什么？')
    st.markdown(
        '在 **1 万元优惠券预算**内，提升 **90 天复购率**。\n'
        '方法主线：**数据审计 → 特征工程 → 六段分层 → A/B 验证 → ROI 试算 → 敏感性分析**。\n'
        '所有指标口径与 `sql/` 练习、`电商用户价值分层_重做版.ipynb` 完全一致。'
    )
    c1, c2, c3 = st.columns(3)
    ab = load('ab_overall.csv')
    lift = round(float(ab.loc[ab.treatment == 1, '复购率'].iloc[0])
                 - float(ab.loc[ab.treatment == 0, '复购率'].iloc[0]), 4)
    c1.metric('整体复购增量', f'{lift * 100:.2f} 个百分点', '对照组 vs 处理组')
    c2.metric('最优群体增量（高值流失）', '+15.75 个百分点', '发券效果显著')
    comp = load('strategy_comparison.csv')
    roi_seg = float(comp.loc[comp['方案'] == '数据驱动六段式', 'ROI'].iloc[0])
    c3.metric('六段式方案 ROI', f'{roi_seg * 100:.1f}%', '传统 RFM 方案为 -57%')
    st.info(
        '**一句话结论**：不要按"谁最有钱"发券（传统 RFM 思路，会亏），'
        '要按"谁被券撬动的空间最大"发券（高值流失群体，157 人，ROI 约 175%）。'
    )

# ===========================================================================
# 页签 1：数据审计
# ===========================================================================
with tab1:
    st.subheader('这页回答：数据能不能用？')
    st.caption('对应 SQL 练习 01_data_audit.sql。审计通过才敢分析，就像做饭前先检查食材有没有坏。')

    audit = load('audit_overview.csv')
    d1, d2, d3 = st.columns([1, 1, 1])
    with d1:
        st.markdown('**表行数**（预期 4000 / 7628 / 27989 / 4000）')
        st.dataframe(audit, hide_index=True, width='stretch')
    with d2:
        st.markdown('**时间范围**（必须落在特征窗口内，否则 = 数据泄漏）')
        st.dataframe(load('date_ranges.csv'), hide_index=True, width='stretch')
        st.markdown('**A/B 分组人数**（应接近 1:1）')
        st.dataframe(load('experiment_balance.csv'), hide_index=True, width='stretch')
    with d3:
        st.markdown('**金额体检**（MIN/AVG/MAX 三联看异常值）')
        st.dataframe(load('order_stats.csv'), hide_index=True, width='stretch')

    cat = load('category_summary.csv')
    fig = px.bar(cat, x='category', y='订单数', text='订单数',
                 title='品类订单分布（8 个品类口径干净，无大小写/别名分裂）')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, width='stretch')
    st.markdown(
        '**怎么读**：审计的三道关——①行数主键没缺 ②日期没越界 ③金额无异常、品类无脏值。'
        '本项目三道关全过，可以进入特征工程。'
    )

# ===========================================================================
# 页签 2：用户分层
# ===========================================================================
with tab2:
    st.subheader('这页回答：用户可以分成哪几类，各有什么特征？')
    st.caption('分层逻辑：R（最近 90 天是否购买）× M 三分位 = 2×3 = 6 个群体。对应 SQL 练习 03。')

    counts = load('segment_counts.csv')
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(counts, x='Segment', y='用户数', color='Segment',
                     color_discrete_map=SEG_COLORS, category_orders={'Segment': SEGMENT_ORDER},
                     text='用户数', title='六段人数分布（最小群体 157 人 > 100，足够稳定）')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with c2:
        prof = load('segment_profile.csv').set_index('Segment').reindex(SEGMENT_ORDER)
        heat = prof[['平均R分', '平均F分', '平均M分', '平均I分', '平均L分']]
        fig = px.imshow(heat, text_auto='.1f', color_continuous_scale='YlGnBu', aspect='auto',
                        title='群体画像热力图（每个群体的平均得分）')
        st.plotly_chart(fig, width='stretch')

    scatter = load('segment_scatter.csv')
    fig = px.scatter(scatter, x='R_score', y='M_score', color='Segment',
                     color_discrete_map=SEG_COLORS, category_orders={'Segment': SEGMENT_ORDER},
                     opacity=0.5, hover_data=['user_id', 'monetary_180d', 'recency_days'],
                     title='R 分 × M 分散点图（分层的几何直观：横轴=最近不最近，纵轴=花不花钱）')
    st.plotly_chart(fig, width='stretch')
    st.markdown(
        '**怎么读**：右上角"高 R 高 M"是核心价值；左上角"低 R 高 M"是高值流失——以前花得多、最近不来了，'
        '是营销预算最该盯的人群。'
    )

# ===========================================================================
# 页签 3：A/B 实验
# ===========================================================================
with tab3:
    st.subheader('这页回答：优惠券到底有没有用？对谁有用？')
    st.caption('核心概念：增量 = 处理组复购率 − 对照组复购率。只看处理组 42% 会高估效果（感冒自愈效应）。对应 SQL 练习 04。')

    ab = load('ab_overall.csv')
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(ab, x='treatment', y='复购率', text='复购率',
                     labels={'treatment': '组别（0=对照 1=处理）'},
                     title='整体复购率对比', color='treatment', color_continuous_scale='RdBu')
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = go.Figure(go.Indicator(
            mode='number+delta', value=float(ab.loc[ab.treatment == 1, '复购率'].iloc[0]) * 100,
            delta={'reference': float(ab.loc[ab.treatment == 0, '复购率'].iloc[0]) * 100,
                   'valueformat': '.2f', 'suffix': ' pp'},
            number={'suffix': ' %'}, title={'text': '整体增量（处理 − 对照）'}))
        st.plotly_chart(fig, width='stretch')

    abs_ = load('ab_by_segment.csv')
    fig = go.Figure()
    fig.add_bar(y=abs_['Segment'], x=abs_['conv_control'], name='对照组（不发券）',
                orientation='h', marker_color='#FF6B6B')
    fig.add_bar(y=abs_['Segment'], x=abs_['conv_treatment'], name='处理组（发券）',
                orientation='h', marker_color='#0B5CAD')
    fig.update_layout(barmode='group', title='分群体复购率：发券 vs 不发券',
                      yaxis={'categoryorder': 'array', 'categoryarray': SEGMENT_ORDER[::-1]})
    st.plotly_chart(fig, width='stretch')

    fig = px.bar(abs_, x='Segment', y='lift_pp', color='Segment',
                 color_discrete_map=SEG_COLORS, category_orders={'Segment': SEGMENT_ORDER},
                 text='lift_pp', title='增量（百分点）：高值流失 +15.75，整体平均只有 +3.1')
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')
    st.markdown(
        '**怎么读**：同样的券，在不同群体上的"药效"差 5 倍以上。'
        '不发券也会复购的核心价值，增量只有 5 个百分点；高值流失则被券明显撬动（+15.75）。'
        '新近低值的增量为负——发券可能打扰了本来会自然复购的人。'
    )

# ===========================================================================
# 页签 4：ROI 与预算
# ===========================================================================
with tab4:
    st.subheader('这页回答：钱该花给谁？花多少？')
    st.caption('ROI = 增量净利润 / 总成本。假设：毛利率 35%，券成本 12 元/张，触达成本 0.5 元/人。对应 SQL 练习 04.3/04.4。')

    roi = load('roi_by_segment.csv')
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(roi, x='Segment', y='增量净利润', color='Segment',
                     color_discrete_map=SEG_COLORS, category_orders={'Segment': SEGMENT_ORDER},
                     text='增量净利润', title='分群体增量净利润（只有高值流失为正）')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.bar(roi, x='Segment', y='ROI', color='Segment',
                     color_discrete_map=SEG_COLORS, category_orders={'Segment': SEGMENT_ORDER},
                     text='ROI', title='分群体 ROI')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    comp = load('strategy_comparison.csv')
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(comp, x='方案', y='增量净利润', text='增量净利润',
                     color='方案', color_discrete_map={'传统 RFM 前 20%': '#FF6B6B', '数据驱动六段式': '#0B5CAD'},
                     title='两种方案增量净利润对比（10000 元预算）')
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.bar(comp, x='方案', y='ROI', text='ROI',
                     color='方案', color_discrete_map={'传统 RFM 前 20%': '#FF6B6B', '数据驱动六段式': '#0B5CAD'},
                     title='两种方案 ROI 对比')
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    st.dataframe(comp.round(2), hide_index=True, width='stretch')
    st.markdown(
        '**怎么读**：传统 RFM 按"谁最有钱"选客，结果选中大量核心价值用户——他们不买券也会复购，'
        '券的核销成本（32% 核销率 × 12 元）直接把利润打成负数（-2000 元）。'
        '六段式只投 158 个高值流失用户，净利润 +1031 元，ROI 188%。'
    )

# ===========================================================================
# 页签 5：敏感性分析
# ===========================================================================
with tab5:
    st.subheader('这页回答：结论扛不扛得住假设变化？')
    st.caption('把毛利率、券成本、核销率、增量效应、客单价分别上下浮动，看 ROI 是否仍为正。对应 notebook 22_sensitivity.py。')

    sens = load('sensitivity.csv')
    fig = px.bar(sens, x='情景', y='ROI', color='ROI', color_continuous_scale='RdYlGn',
                 text='ROI', range_y=[0, max(sens['ROI']) * 1.15],
                 title='各情景下的 ROI（所有悲观情景下 ROI 仍 > 0）')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')

    worst = sens.loc[sens['ROI'].idxmin()]
    c1, c2 = st.columns(2)
    c1.metric('最差情景', worst['情景'], f"ROI {worst['ROI']:.2f}")
    c2.metric('基准情景 ROI', f"{sens.loc[sens['情景'] == '增量效应×1.00', 'ROI'].iloc[0]:.2f}", '结论稳健')
    st.markdown(
        '**怎么读**：即使增量效果打五折（增量效应×0.50），ROI 仍有 0.44，不亏钱。'
        '这说明"优先投高值流失"的结论不依赖于某个精确的假设值。'
    )

# ===========================================================================
# 页签 6：结论与提醒
# ===========================================================================
with tab6:
    st.subheader('结论与诚实声明')
    st.markdown(
        '- ✅ **结论 1**：六段分层中，高值流失（157 人）是唯一发券能赚钱的群体。\n'
        '- ✅ **结论 2**：核心价值用户不该发券——增量小、核销成本高。\n'
        '- ✅ **结论 3**：在悲观假设下结论依然成立（敏感性分析通过）。\n'
        '- ⚠️ **提醒 1**：所有数据为模拟数据，数字仅作教学演示，不代表真实业务。\n'
        '- ⚠️ **提醒 2**：高值流失群体仅 157 人，其中对照组 73 人，统计功效有限，实操前应扩大实验样本。\n'
        '- ⚠️ **提醒 3**：p 值计算与特征验证留在 Python（notebook 第 6 节），看板只展示描述性结论。'
    )
    st.markdown(
        '**面试/汇报时可以这样收尾**：'
        '"我不只做了分层，还用 A/B 实验验证了券的真实增量，并用 ROI 和敏感性分析证明了'
        '投高值流失这个决策在假设波动下依然稳健——这比只报一个 RFM 分层结论可靠得多。"'
    )
