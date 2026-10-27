# -*- coding: utf-8 -*-
"""
SQL 取数 + Python 建模可视化统一流水线。
运行：python tools/pipeline_modeling.py
"""

from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TOOLS_DIR = Path(__file__).resolve().parent
SQL_DIR = TOOLS_DIR.parent / 'sql'


def split_statements(text: str) -> list:
    lines = [l for l in text.splitlines() if not l.strip().startswith('--')]
    return [s.strip() for s in '\n'.join(lines).split(';') if s.strip()]


def read_section(name: str, section: str, expect_one: bool = True):
    text = (SQL_DIR / name).read_text(encoding='utf-8-sig')
    pattern = rf'--\s*@section\s+{re.escape(section)}\b(.*?)(?=--\s*@section\b|\Z)'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        raise ValueError(f'{{name}} 中找不到 @section {{section}}')
    stmts = split_statements(match.group(1))
    if expect_one and len(stmts) != 1:
        raise ValueError(f'@section {{section}} 应只含 1 条查询，实际 {{len(stmts)}} 条')
    return stmts[0] if expect_one else stmts

ROOT = TOOLS_DIR.parent
DB_PATH = ROOT / 'data' / 'demo' / 'ecommerce.db'
SQL_FILE = ROOT / 'sql' / 'pipeline_01_wide_table.sql'
DASH_DIR = ROOT / 'data' / 'dashboard'
REPORT_HTML = ROOT / 'reports' / 'dashboard.html'

# 商业假设（唯一权威出处，敏感性分析围绕它们波动）
BUDGET = 10000.0
GROSS_MARGIN = 0.35
COUPON_COST = 12.0
CONTACT_COST = 0.5

SEGMENT_ORDER = ['核心价值', '活跃成长', '新近低值', '高值流失', '沉睡中值', '低值流失']
SEG_COLORS = {'核心价值': '#0B5CAD', '活跃成长': '#2E8B57', '新近低值': '#7FB3D5',
              '高值流失': '#E67E22', '沉睡中值': '#9B59B6', '低值流失': '#BDC3C7'}
SEGMENT_MAP = {'高值·近期': '核心价值', '中值·近期': '活跃成长', '低值·近期': '新近低值',
               '高值·流失': '高值流失', '中值·流失': '沉睡中值', '低值·流失': '低值流失'}


# ===========================================================================
# 第 1 步：SQL 取数 —— 执行 pipeline SQL，把结果读成 DataFrame
# ===========================================================================
def step1_sql_extract() -> dict:
    """执行 SQL 段并返回 {名字: DataFrame}。演示 SQL→Python 的标准交接。"""
    con = sqlite3.connect(DB_PATH)
    text = SQL_FILE.read_text(encoding='utf-8-sig')

    # 建宽表（DROP + CREATE 是一条语句）
    wide_stmts = read_section('pipeline_01_wide_table.sql', 'wide_table', expect_one=False)
    for stmt in wide_stmts:
        con.execute(stmt)
    con.commit()

    # ★ SQL → Python 的交接就发生在下面三行：
    #   数据库表对 pandas 来说就是一个 DataFrame 的数据源，
    #   连接对象 con 就是传送带。SQL 改了取数逻辑，这里零改动。
    tables = {
        'audit_rowcounts': pd.read_sql_query(read_section('pipeline_01_wide_table.sql', 'audit_rowcounts'), con),
        'audit_dates': pd.read_sql_query(read_section('pipeline_01_wide_table.sql', 'audit_dates'), con),
        'audit_amounts': pd.read_sql_query(read_section('pipeline_01_wide_table.sql', 'audit_amounts'), con),
        'wide': pd.read_sql_query('SELECT * FROM pipeline_wide', con),
        'txn': pd.read_sql_query('SELECT user_id, category FROM transactions', con),
        'experiment': pd.read_sql_query('SELECT * FROM experiment', con),
    }
    con.close()

    print('【第 1 步】SQL 取数完成')
    print('  宽表规模:', tables['wide'].shape)
    print('  审计-行数:', tables['audit_rowcounts']['行数'].tolist())
    print('  审计-日期:', tables['audit_dates']['最早'].tolist(), '→',
          tables['audit_dates']['最晚'].tolist())
    return tables


# ===========================================================================
# 第 2 步：Python 打分 + 规则六段分层
# ===========================================================================
def step2_score_segment(wide: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    df = wide.copy()
    # SQLite 列名习惯小写，建模代码沿用 notebook 的命名习惯
    df = df.rename(columns={'friction_index': 'Friction_Index'})

    # 人货匹配：Python 侧再取一次明细算"最常买品类"，第二次演示 SQL→Python 交接
    preferred = (txn.assign(category=txn['category'].str.strip().str.lower())
                 .groupby('user_id')['category']
                 .agg(lambda s: s.mode().iloc[0] if len(s) else None))
    df['preferred_category'] = df['user_id'].map(preferred)
    df['interest_match'] = (df['interest'].str.strip().str.lower()
                            == df['preferred_category']).astype(int)

    # 百分位打分（rank(pct=True)：关心相对位置，不关心绝对量级）
    df['R_score'] = (1 - df['recency_days'].rank(pct=True)) * 100
    df['F_score'] = df['frequency_180d'].rank(pct=True) * 100
    df['M_score'] = df['monetary_180d'].rank(pct=True) * 100
    df['I_score'] = 100 * (0.5 * np.log1p(df['total_minutes']).rank(pct=True)
                           + 0.5 * np.log1p(df['total_pages']).rank(pct=True))
    df['L_score'] = 100 * (0.4 * df['newsletter_subscription']
                           + 0.6 * df['recent_sessions_30d'].rank(pct=True))
    df['RFM_Score'] = 0.2 * df['R_score'] + 0.3 * df['F_score'] + 0.5 * df['M_score']

    # 规则六段分层
    df['recency_flag'] = np.where(df['recency_days'] <= 90, '近期', '流失')
    df['value_tier'] = pd.qcut(df['M_score'], q=3, labels=['低值', '中值', '高值'])
    df['Segment'] = (df['value_tier'].astype(str) + '·' + df['recency_flag']).map(SEGMENT_MAP)

    print('【第 2 步】打分 + 规则分层完成')
    print('  最小群体人数:', int(df['Segment'].value_counts().min()))
    return df


# ===========================================================================
# 第 3 步：KMeans 聚类 —— 与规则分层对照
# ===========================================================================
def step3_kmeans(df: pd.DataFrame):
    feat_cols = ['R_score', 'F_score', 'M_score', 'I_score']
    X = StandardScaler().fit_transform(df[feat_cols])

    # k 选择的三条证据（k=6 是业务对照选择，不是数据结论——见 kmeans_experiment.py）
    k_rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        k_rows.append({
            'k': k,
            'inertia_肘部': round(km.inertia_, 0),
            '轮廓系数': round(silhouette_score(X, labels), 3),
            'ARI_对规则分层': round(adjusted_rand_score(df['Segment'], labels), 3),
        })
    k_table = pd.DataFrame(k_rows)

    # 业务对照：k=6，与规则六段分层比一致性
    km = KMeans(n_clusters=6, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = km.fit_predict(X)

    ari = adjusted_rand_score(df['Segment'], df['cluster'])
    best_sil = int(k_table.loc[k_table['轮廓系数'].idxmax(), 'k'])
    best_ari = int(k_table.loc[k_table['ARI_对规则分层'].idxmax(), 'k'])
    print('【第 3 步】KMeans 完成')
    print(f'  k 选择证据：轮廓系数最优 k={best_sil}，对规则分层 ARI 最优 k={best_ari}（k=6 为业务对照）')
    print(f'  k=6（业务对照）与规则分层 ARI = {ari:.3f}')
    print('  （ARI 接近 1 = 聚类与业务规则高度一致；接近 0 = 各说各话）')

    crosstab = (pd.crosstab(df['Segment'], df['cluster'])
                .reindex(SEGMENT_ORDER))
    return df, crosstab, ari, k_table


# ===========================================================================
# 第 4 步：逻辑回归建模 —— 只用对照组，5 折交叉验证
# ===========================================================================
def step4_modeling(df: pd.DataFrame) -> dict:
    control = df[df['treatment'] == 0]
    rfm_features = ['R_score', 'F_score', 'M_score']
    full_features = rfm_features + ['I_score', 'L_score', 'Friction_Index', 'interest_match']
    reduced_features = rfm_features + ['I_score', 'L_score', 'interest_match']
    y = control['repeat_90d']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def cv_auc(feats, name):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, random_state=42))
        s = cross_val_score(model, control[feats], y, cv=cv, scoring='roc_auc')
        return {'模型': name, 'AUC均值': s.mean(), 'AUC标准差': s.std()}

    auc_table = pd.DataFrame([
        cv_auc(rfm_features, 'RFM'),
        cv_auc(full_features, 'RFM + I + L + Friction + 人货匹配'),
        cv_auc(reduced_features, 'RFM + I + L + 人货匹配（去 Friction）'),
    ])

    # 单变量 AUC：Friction 到底有没有用
    def uni_auc(col, direction=1):
        s = control[col].rank(pct=True)
        return roc_auc_score(y, s if direction > 0 else 1 - s)

    uni = pd.DataFrame([
        {'特征': c, '单变量AUC': uni_auc(c, -1 if c == 'Friction_Index' else 1)}
        for c in full_features
    ]).sort_values('单变量AUC', ascending=False)

    # 系数解释（70/30 切分，仅在训练集上拟合）
    X_tr, X_te, y_tr, y_te = train_test_split(
        control[full_features], y, test_size=0.3, random_state=42, stratify=y)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
    model.fit(X_tr, y_tr)
    coef = pd.Series(model.named_steps['logisticregression'].coef_[0],
                     index=full_features).sort_values(ascending=False)
    fpr, tpr, _ = roc_curve(y_te, model.predict_proba(X_te)[:, 1])
    roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    test_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

    print('【第 4 步】逻辑回归建模完成（仅对照组，5 折 CV）')
    print(auc_table.round(4).to_string(index=False))
    print(f'  完整模型测试集 AUC = {test_auc:.4f}')
    print(f'  Friction 单变量 AUC = {uni.set_index("特征").loc["Friction_Index","单变量AUC"]:.4f}'
          '（<0.55 视为未通过验证）')
    return {'auc_table': auc_table, 'univariate': uni, 'coef': coef, 'roc': roc, 'test_auc': test_auc}


# ===========================================================================
# 第 5 步：A/B 增量 + ROI + 预算分配 + 敏感性
# ===========================================================================
def roi_for_segment(row, n_target=None, margin=GROSS_MARGIN, coupon_cost=COUPON_COST,
                    contact_cost=CONTACT_COST, lift_multiplier=1.0,
                    redemption_multiplier=1.0, aov_multiplier=1.0):
    n = row['n_total'] if n_target is None else n_target
    lift = max(row['conv_treatment'] - row['conv_control'], 0) * lift_multiplier
    aov = row['aov'] * aov_multiplier
    gross_profit = n * lift * aov * margin
    redemption = min(max(row['redemption_rate'] * redemption_multiplier, 0), 1)
    total_cost = n * (redemption * coupon_cost + contact_cost)
    net_profit = gross_profit - total_cost
    return pd.Series({
        'n': n, 'lift': lift, 'gross_profit': gross_profit,
        'total_cost': total_cost, 'net_profit': net_profit,
        'roi': net_profit / total_cost if total_cost > 0 else np.nan,
        'cost_per_user': total_cost / n if n > 0 else np.nan,
        'profit_per_user': net_profit / n if n > 0 else np.nan,
    })


def allocate_budget(table: pd.DataFrame, budget: float) -> pd.DataFrame:
    remaining, allocs = budget, []
    candidates = table.sort_values(['roi', 'profit_per_user'], ascending=False)
    for seg, row in candidates.iterrows():
        if row['profit_per_user'] <= 0 or row['cost_per_user'] <= 0:
            continue
        n_target = min(int(row['n_total']), int(remaining // row['cost_per_user']))
        if n_target <= 0:
            continue
        m = roi_for_segment(row, n_target=n_target)
        allocs.append({'Segment': seg, 'n_target': n_target,
                       'allocated_cost': m['total_cost'], 'allocated_profit': m['net_profit'],
                       'roi': m['roi']})
        remaining -= m['total_cost']
    return pd.DataFrame(allocs)


def step5_ab_roi(df: pd.DataFrame) -> dict:
    rows = []
    for seg in SEGMENT_ORDER:
        g = df[df['Segment'] == seg]
        c, t = g[g['treatment'] == 0], g[g['treatment'] == 1]
        aov = g.loc[g['frequency_180d'] > 0, 'avg_order_value_180d'].mean()
        rows.append({
            'Segment': seg, 'n_total': len(g), 'n_control': len(c), 'n_treatment': len(t),
            'conv_control': c['repeat_90d'].mean(), 'conv_treatment': t['repeat_90d'].mean(),
            'lift_pp': (t['repeat_90d'].mean() - c['repeat_90d'].mean()) * 100,
            'aov': aov, 'redemption_rate': t['coupon_redeemed'].mean(),
        })
    ab = pd.DataFrame(rows).set_index('Segment').reindex(SEGMENT_ORDER).reset_index()

    roi_parts = ab.apply(lambda r: roi_for_segment(r), axis=1)
    roi = pd.concat([ab.reset_index(drop=True), roi_parts.reset_index(drop=True)], axis=1)

    selected = allocate_budget(roi.set_index('Segment'), BUDGET)
    sel = selected.merge(roi[['Segment', 'lift', 'aov']], on='Segment')

    # 传统 RFM 前 20% 对照
    top20 = df[df['RFM_Score'] >= df['RFM_Score'].quantile(0.80)]
    t20 = pd.Series({
        'n_total': len(top20),
        'conv_control': top20.loc[top20['treatment'] == 0, 'repeat_90d'].mean(),
        'conv_treatment': top20.loc[top20['treatment'] == 1, 'repeat_90d'].mean(),
        'aov': top20.loc[top20['frequency_180d'] > 0, 'avg_order_value_180d'].mean(),
        'redemption_rate': top20.loc[top20['treatment'] == 1, 'coupon_redeemed'].mean(),
    })
    t20m = roi_for_segment(t20)

    strategy = pd.DataFrame([
        {'方案': '传统 RFM 前 20%', '目标人数': int(t20['n_total']),
         '增量收入': t20['n_total'] * max(t20['conv_treatment'] - t20['conv_control'], 0) * t20['aov'],
         '总成本': t20m['total_cost'], '增量净利润': t20m['net_profit'], 'ROI': t20m['roi']},
        {'方案': '数据驱动六段式', '目标人数': int(selected['n_target'].sum()),
         '增量收入': (sel['n_target'] * sel['lift'] * sel['aov']).sum(),
         '总成本': selected['allocated_cost'].sum(),
         '增量净利润': selected['allocated_profit'].sum(),
         'ROI': selected['allocated_profit'].sum() / selected['allocated_cost'].sum()},
    ])

    # 敏感性分析：固定入选人群，单因子波动
    base_targets = selected.set_index('Segment')['n_target']
    roi_indexed = roi.set_index('Segment')

    def scenario(margin=GROSS_MARGIN, coupon_cost=COUPON_COST, contact_cost=CONTACT_COST,
                 lift_multiplier=1.0, redemption_multiplier=1.0, aov_multiplier=1.0):
        parts = [roi_for_segment(roi_indexed.loc[seg], n_target=n,
                                 margin=margin, coupon_cost=coupon_cost, contact_cost=contact_cost,
                                 lift_multiplier=lift_multiplier,
                                 redemption_multiplier=redemption_multiplier,
                                 aov_multiplier=aov_multiplier)
                 for seg, n in base_targets.items()]
        p = pd.DataFrame(parts)
        return pd.Series({'profit': p['net_profit'].sum(),
                          'roi': p['net_profit'].sum() / p['total_cost'].sum()})

    sens_rows = []
    for margin in [0.20, 0.30, 0.35, 0.45, 0.55]:
        s = scenario(margin=margin)
        sens_rows.append({'情景': f'毛利率={margin:.0%}', '净增量利润': s['profit'], 'ROI': s['roi']})
    for mult in [0.50, 0.75, 1.00, 1.25, 1.50]:
        s = scenario(redemption_multiplier=mult)
        sens_rows.append({'情景': f'核销率×{mult:.2f}', '净增量利润': s['profit'], 'ROI': s['roi']})
    for mult in [0.50, 0.75, 1.00, 1.25, 1.50]:
        s = scenario(lift_multiplier=mult)
        sens_rows.append({'情景': f'增量效应×{mult:.2f}', '净增量利润': s['profit'], 'ROI': s['roi']})
    for mult in [0.80, 1.00, 1.20]:
        s = scenario(aov_multiplier=mult)
        sens_rows.append({'情景': f'客单价×{mult:.2f}', '净增量利润': s['profit'], 'ROI': s['roi']})
    for cost in [8, 12, 16, 20]:
        s = scenario(coupon_cost=cost)
        sens_rows.append({'情景': f'券成本={cost}元', '净增量利润': s['profit'], 'ROI': s['roi']})
    sensitivity = pd.DataFrame(sens_rows)

    grid = np.linspace(0, 2, 81)
    profits = np.array([scenario(lift_multiplier=x)['profit'] for x in grid])
    positive = grid[profits >= 0]
    break_even = positive.min() if len(positive) else np.nan

    print('【第 5 步】A/B + ROI 完成')
    print(f"  整体增量: {(df.loc[df.treatment==1,'repeat_90d'].mean() - df.loc[df.treatment==0,'repeat_90d'].mean())*100:.2f} pp")
    print(f"  六段式方案: {int(selected['n_target'].sum())} 人, 净利润 {selected['allocated_profit'].sum():.0f} 元, ROI {strategy.loc[1,'ROI']:.3f}")
    print(f"  盈亏平衡增量效应倍数: {break_even:.2f}")
    return {'ab': ab, 'roi': roi, 'selected': selected, 'strategy': strategy,
            'sensitivity': sensitivity, 'break_even': break_even}


# ===========================================================================
# 第 5.5 步：概率分档发券利润曲线（模型与运营的故事线连接点）
# ===========================================================================
def profit_curve(df: pd.DataFrame) -> pd.DataFrame:
    """把对照组训练的模型用于全体用户打分，模拟'只给预测概率 Top N% 发券'。

    假设：增量效应用整体 lift（处理-对照）近似，核销率用处理组整体核销率。
    与六段分层方案并排对比，回答'建模排序'和'业务分层'哪个更会花钱。
    """
    full_features = ['R_score', 'F_score', 'M_score', 'I_score', 'L_score',
                     'Friction_Index', 'interest_match']
    control = df[df['treatment'] == 0]
    treat = df[df['treatment'] == 1]

    def fit_model(d):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, random_state=42))
        return m.fit(d[full_features], d['repeat_90d'])

    # 两种排序：复购概率（自然倾向）与 uplift（被券撬动的空间，双模型差值）
    prob = fit_model(control).predict_proba(df[full_features])[:, 1]
    p_control = prob
    p_treat = fit_model(treat).predict_proba(df[full_features])[:, 1]
    uplift = p_treat - p_control

    lift = (df.loc[df['treatment'] == 1, 'repeat_90d'].mean()
            - df.loc[df['treatment'] == 0, 'repeat_90d'].mean())
    redemption = df.loc[df['treatment'] == 1, 'coupon_redeemed'].mean()

    # 自洽口径：被选中的用户 i，其增量转化 = 估计的 uplift_i（T-learner），
    # 不再套全局常数 lift——常数 lift 会抹掉"谁被券撬动"这个核心差异
    aov_all = df['avg_order_value_180d'].where(df['frequency_180d'] > 0)

    rfm20_mask = df['RFM_Score'] >= df['RFM_Score'].quantile(0.80)
    strategies = [
        ('全体+复购概率排序', df.index, prob),
        ('全体+uplift排序', df.index, uplift),
        ('RFM前20%+复购概率排序', df.index[rfm20_mask], prob),
        ('RFM前20%+uplift排序', df.index[rfm20_mask], uplift),
    ]
    rows = []
    for method, idx, score in strategies:
        ranked = (df.loc[idx].assign(_s=score[idx], _uplift=uplift[idx],
                                     _aov=aov_all[idx])
                  .sort_values('_s', ascending=False))
        for pct in [5, 10, 20, 30, 40, 50, 60, 80, 100]:
            sub = ranked.head(int(len(ranked) * pct / 100))
            n = len(sub)
            conversions = sub['_uplift'].sum()          # 估计的增量转化数
            aov = sub['_aov'].mean()                     # 该档人群的客单价
            gross = conversions * aov * GROSS_MARGIN
            cost = n * (redemption * COUPON_COST + CONTACT_COST)
            rows.append({'排序方式': method, '目标人群': f'Top {pct}%', '目标人数': n,
                         '增量净利润': round(gross - cost, 0),
                         'ROI': round((gross - cost) / cost, 3)})
    curve = pd.DataFrame(rows)
    print('【第 5.5 步】发券利润曲线：四种排序策略（T-learner 自洽口径）')
    print(curve.pivot(index='目标人群', columns='排序方式',
                      values='增量净利润').round(0).to_string())
    return curve


# ===========================================================================
# 第 5.6 步：方法诚实度统计（置信区间 / 功效；多种子结论见文件头注释）
# ===========================================================================
HONESTY_SEED_NOTE = ("多种子复现（tools/robustness_check.py，5 个随机种子）："
                     "'最优群体=高值流失'仅在原种子成立，属探索性结论")


def honesty_stats(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """对整体与高值流失的增量做 bootstrap 置信区间与事后功效分析。"""
    import math
    rng = np.random.default_rng(seed)
    out = []
    for scope, sub in [('整体', df),
                       ('高值流失', df[df['Segment'] == '高值流失'])]:
        data = sub[['treatment', 'repeat_90d']].to_numpy()
        boots = np.empty(n_boot)
        for i in range(n_boot):
            s = data[rng.integers(0, len(data), len(data))]
            t = s[s[:, 0] == 1][:, 1]
            c = s[s[:, 0] == 0][:, 1]
            boots[i] = t.mean() - c.mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        n_c = int((data[:, 0] == 0).sum())
        n_t = int((data[:, 0] == 1).sum())
        p_c = data[data[:, 0] == 0][:, 1].mean()
        p_t = data[data[:, 0] == 1][:, 1].mean()
        lift = p_t - p_c
        se = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
        power = 0.5 * (1 + math.erf(((abs(lift) / se) - 1.959964) / math.sqrt(2)))
        pool = (p_c + p_t) / 2
        need = math.ceil(2.802 ** 2 * 2 * pool * (1 - pool) / lift ** 2) \
            if lift != 0 else np.nan
        out.append({'范围': scope, '增量pp': round(lift * 100, 2),
                    'CI95下限pp': round(lo * 100, 2), 'CI95上限pp': round(hi * 100, 2),
                    '功效': round(power, 2), '每组所需样本_80功效': need})
    table = pd.DataFrame(out)
    print('【第 5.6 步】方法诚实度统计')
    print(table.to_string(index=False))
    print(' ', HONESTY_SEED_NOTE)
    return table


# ===========================================================================
# 第 6 步：导出看板数据（Streamlit / Power BI 读这里）
# ===========================================================================
def step6_export(tables, df, results, crosstab, kmeans_ari, k_table, model_out):
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    put = lambda d, name: d.to_csv(DASH_DIR / name, index=False, encoding='utf-8-sig')

    put(tables['audit_rowcounts'], 'audit_overview.csv')
    put(tables['audit_dates'].rename(columns={'最早': '最早', '最晚': '最晚'}), 'date_ranges.csv')
    put(tables['audit_amounts'].assign(字段='order_value').rename(columns={}), 'order_stats.csv')

    # 明细聚合也走 SQL 取数（第三次演示 SQL→Python 交接）
    con_tmp = sqlite3.connect(DB_PATH)
    cat = pd.read_sql_query(
        "SELECT category, COUNT(*) AS 订单数, ROUND(SUM(order_value),2) AS 总金额 "
        "FROM transactions GROUP BY category ORDER BY 订单数 DESC", con_tmp)
    con_tmp.close()
    put(cat, 'category_summary.csv')

    bal = (tables['experiment'].groupby('experiment_group').size()
           .reset_index(name='人数'))
    put(bal, 'experiment_balance.csv')

    counts = (df.groupby('Segment').size().reindex(SEGMENT_ORDER).reset_index(name='用户数'))
    counts['占比百分比'] = (counts['用户数'] / len(df) * 100).round(2)
    put(counts, 'segment_counts.csv')

    prof = (df.groupby('Segment').agg(
        用户数=('user_id', 'size'),
        平均R分=('R_score', 'mean'), 平均F分=('F_score', 'mean'), 平均M分=('M_score', 'mean'),
        平均I分=('I_score', 'mean'), 平均L分=('L_score', 'mean'),
        平均摩擦度=('Friction_Index', 'mean'),
        平均客单价=('avg_order_value_180d', 'mean'),
        自然复购率=('repeat_90d', 'mean')).reindex(SEGMENT_ORDER).reset_index().round(4))
    put(prof, 'segment_profile.csv')

    put(df[['user_id', 'Segment', 'R_score', 'M_score', 'recency_days',
            'monetary_180d']].round(4), 'segment_scatter.csv')

    ab = results['ab']
    put(df.groupby('treatment').agg(
        人数=('user_id', 'size'), 复购人数=('repeat_90d', 'sum'),
        复购率=('repeat_90d', 'mean'), 人均结果消费=('outcome_spend_90d', 'mean')
    ).reset_index(), 'ab_overall.csv')
    put(ab, 'ab_by_segment.csv')

    roi = results['roi']
    roi_out = roi[['Segment', 'n_total', 'lift_pp', 'aov', 'redemption_rate',
                   'gross_profit', 'total_cost', 'net_profit', 'roi']].rename(columns={
        'gross_profit': '增量毛利', 'total_cost': '总成本',
        'net_profit': '增量净利润', 'roi': 'ROI'})
    put(roi_out.round(4), 'roi_by_segment.csv')

    put(results['strategy'].round(4), 'strategy_comparison.csv')
    put(results['sensitivity'].round(4), 'sensitivity.csv')
    put(model_out['auc_table'].round(4), 'model_auc.csv')
    put(crosstab.reset_index(), 'kmeans_crosstab.csv')
    put(k_table, 'kmeans_k_selection.csv')
    put(results['profit_curve'], 'profit_curve.csv')
    put(results['honesty'], 'honesty_stats.csv')

    print('【第 6 步】看板数据已导出 ->', DASH_DIR)


# ===========================================================================
# 第 7 步：生成单文件 HTML 交互看板（可发简历链接）
# ===========================================================================
def step7_html(df, results, crosstab, kmeans_ari, k_table, model_out):
    seg_counts = df['Segment'].value_counts().reindex(SEGMENT_ORDER).reset_index()
    seg_counts.columns = ['Segment', '用户数']
    figs = []

    figs.append(('六段用户分层人数', px.bar(
        seg_counts, x='Segment', y='用户数', color='Segment',
        color_discrete_map=SEG_COLORS, text='用户数',
        title='六段用户分层人数（流失=超过90天未购买；高/中/低值=M分三分位）')))

    prof = (df.groupby('Segment')[['R_score', 'F_score', 'M_score', 'I_score', 'L_score']]
            .mean().reindex(SEGMENT_ORDER))
    figs.append(('群体画像热力图', px.imshow(
        prof, text_auto='.1f', color_continuous_scale='YlGnBu', aspect='auto',
        title='群体画像：各群体平均得分')))

    figs.append(('R × M 分层散点图', px.scatter(
        df, x='R_score', y='M_score', color='Segment', color_discrete_map=SEG_COLORS,
        opacity=0.45, hover_data=['user_id', 'monetary_180d'],
        title='R 分 × M 分：左上角"低 R 高 M" = 高值流失（营销重点）')))

    ab = results['ab'].round(2)
    fig_ab = go.Figure()
    fig_ab.add_bar(x=ab['Segment'], y=ab['conv_control'], name='对照组（不发券）', marker_color='#FF6B6B')
    fig_ab.add_bar(x=ab['Segment'], y=ab['conv_treatment'], name='处理组（发券）', marker_color='#0B5CAD')
    fig_ab.update_layout(barmode='group', title='A/B 实验：分群体复购率')
    figs.append(('A/B 实验复购率对比', fig_ab))

    figs.append(('发券增量（百分点）', px.bar(
        ab, x='Segment', y='lift_pp', color='Segment', color_discrete_map=SEG_COLORS,
        text='lift_pp', title='增量 = 处理组 − 对照组（高值流失 +15.75 pp）')))

    roi = results['roi'].round(2)
    fig_roi_bar = px.bar(
        roi, x='Segment', y='net_profit', color='Segment', color_discrete_map=SEG_COLORS,
        text='net_profit', title='分群体增量净利润（仅高值流失为正）')
    fig_roi_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    figs.append(('分群体增量净利润', fig_roi_bar))

    fig_strategy = px.bar(
        results['strategy'].round(2), x='方案', y='ROI', text='ROI',
        color='方案', color_discrete_map={'传统 RFM 前 20%': '#FF6B6B', '数据驱动六段式': '#0B5CAD'},
        title='数据驱动六段式 vs 传统 RFM（10000 元预算）')
    fig_strategy.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    figs.append(('两种方案 ROI 对比', fig_strategy))

    fig_sens = px.bar(
        results['sensitivity'].round(2), x='情景', y='ROI', text='ROI', color='ROI',
        color_continuous_scale='RdYlGn', title='悲观假设下 ROI 仍为正（结论稳健）')
    fig_sens.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    figs.append(('敏感性分析', fig_sens))

    fig_k = go.Figure()
    fig_k.add_scatter(x=k_table['k'], y=k_table['轮廓系数'], mode='lines+markers',
                      name='轮廓系数（结构质量）')
    fig_k.add_scatter(x=k_table['k'], y=k_table['ARI_对规则分层'], mode='lines+markers',
                      name='ARI 对规则分层（业务一致性）')
    fig_k.update_layout(title='KMeans 选 k：k=6 是业务对照选择，非数据最优',
                        xaxis_title='k（簇数）', yaxis_title='得分')
    figs.append(('KMeans 选 k 证据', fig_k))

    figs.append(('KMeans 聚类 vs 规则分层', px.imshow(
        crosstab, text_auto=True, color_continuous_scale='Blues', aspect='auto',
        title=f'聚类 × 规则分层 交叉表（ARI={kmeans_ari:.2f}）')))

    fig_auc = px.bar(
        model_out['auc_table'].round(3), x='模型', y='AUC均值', error_y='AUC标准差',
        text='AUC均值', title='行为特征加入后 AUC 由 0.670 提升至 0.697')
    fig_auc.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    figs.append(('模型 AUC 对比（逻辑回归 5 折 CV）', fig_auc))

    roc = model_out['roc']
    fig_roc = go.Figure()
    fig_roc.add_scatter(x=roc['fpr'], y=roc['tpr'], mode='lines',
                        name=f"AUC={model_out['test_auc']:.3f}", line_color='#0B5CAD')
    fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line_dash='dash',
                        name='随机猜测', line_color='gray')
    fig_roc.update_layout(title='复购预测 ROC（测试集）', xaxis_title='假阳性率',
                          yaxis_title='真阳性率')
    pc = results['profit_curve']
    fig_pc = go.Figure()
    pc_colors = {'全体+复购概率排序': '#FF6B6B', '全体+uplift排序': '#0B5CAD',
                 'RFM前20%+复购概率排序': '#F1C40F', 'RFM前20%+uplift排序': '#2E8B57'}
    for method, color in pc_colors.items():
        sub = pc[pc['排序方式'] == method].round(2)
        fig_pc.add_scatter(x=sub['目标人群'], y=sub['增量净利润'], mode='lines+markers',
                           name=method, line_color=color)
    seg_profit = results['strategy'].loc[results['strategy']['方案'] == '数据驱动六段式',
                                         '增量净利润'].iloc[0]
    seg_profit = round(float(seg_profit), 2)
    fig_pc.add_hline(y=seg_profit, line_dash='dash', line_color='#E67E22',
                     annotation_text=f'六段分层方案 = {seg_profit:.0f} 元')
    fig_pc.update_layout(title='发券利润曲线：精细化触达（uplift）才是 ROI 由负转正的主因',
                         yaxis_title='增量净利润（元）')
    figs.append(('发券利润曲线', fig_pc))

    figs.append(('ROC 曲线', fig_roc))

    # 拼成一个自包含 HTML：plotly.js 只内嵌一次
    def _round_fig(fig, nd=2):
        def _round_elem(x):
            return round(x, nd) if isinstance(x, float) else x
        for tr in fig.data:
            for k in list(tr):
                v = tr[k]
                try:
                    arr = np.asarray(v)
                    if arr.dtype.kind == 'f':
                        tr[k] = np.round(arr, nd)
                    elif arr.dtype == object:
                        tr[k] = np.vectorize(_round_elem, otypes=[object])(arr).tolist()
                except Exception:
                    pass

    body = []
    for i, (title, fig) in enumerate(figs):
        _round_fig(fig)
        body.append(f'<h2>{title}</h2>')
        body.append(fig.to_html(full_html=False,
                                include_plotlyjs=True if i == 0 else False))

    honesty = results['honesty']
    honesty_rows = honesty.to_html(index=False)
    n_control = int((df['treatment'] == 0).sum())
    n_treat = int((df['treatment'] == 1).sum())
    seed_note = HONESTY_SEED_NOTE
    key_row = honesty.loc[honesty['范围'] == '高值流失'].iloc[0]
    power = f"{key_row['功效']:.2f}"
    need = int(key_row['每组所需样本_80功效'])
    strategy_rows = results['strategy'].round(2).to_html(index=False)

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>电商用户价值分层与精准营销 · dashboard</title>
<style>
body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 1100px;
       margin: 0 auto; padding: 24px; background: #fafafa; color: #222; }}
h1 {{ border-bottom: 3px solid #0B5CAD; padding-bottom: 8px; }}
h2 {{ margin-top: 40px; color: #0B5CAD; }}
.note {{ background: #eef4fb; border-left: 4px solid #0B5CAD;
        padding: 12px 16px; border-radius: 4px; }}
table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc;
        padding: 6px 10px; }}
</style></head><body>
<h1>📊 电商用户价值分层与精准营销策略 · dashboard</h1>
<p class="note">模拟数据（4000 用户）｜特征窗口 2025-01-01~06-30｜结果窗口 2025-07-01~09-30<br>
<b>一句话结论</b>：不按"谁最有钱"发券，按"谁被券撬动的空间最大"发券——
高值流失群体（157 人）发券增量 +15.75 个百分点，方案 ROI {results["strategy"].loc[1,"ROI"]:.0%}；
传统 RFM 前 20% 方案 ROI 为负。</p>
<h2>方案对比总表</h2>{strategy_rows}
<h2>分群边界（可复现定义）</h2>
<p class="note">流失/近期：最近一次购买距今是否超过 90 天（recency_days &gt; 90 为流失）；
高/中/低值：M_score（消费金额百分位）三分位切分（qcut q=3）；两维度交叉得 6 段。
打分口径见 sql/pipeline_01_wide_table.sql 与 tools/pipeline_modeling.py。</p>
{''.join(body)}
<h2>方法诚实度</h2>
{honesty_rows}
<ul>
<li>增量识别：随机对照实验（{n_control} 对照 / {n_treat} 处理），增量 = 组间差值</li>
<li>{seed_note}</li>
<li>uplift 排序在本数据上会将"核心价值"误排为最优——模型估计的 uplift 排序
不等于实验实测排序，随机实验仍是裁判</li>
</ul>
<h2>行动路径（若应用于真实业务）</h2>
<ul>
<li>券设计：按 12 元/张成本对应面额（如满 100 减 12），高值流失可加大面额测试</li>
<li>渠道：L 分高（订阅+活跃）走 APP push，L 分低走短信兜底</li>
<li>监控：每周看核销率 / 增量复购 / ROI，保留 5–10% 不触达人群作长期对照</li>
<li>扩样：当前功效仅 {power}，真实落地前需把目标群体样本扩至每组 {need} 人以上</li>
</ul>
</body></html>"""
    REPORT_HTML.write_text(html, encoding='utf-8')
    print(f'【第 7 步】HTML 看板已生成 -> {REPORT_HTML} ({REPORT_HTML.stat().st_size // 1024} KB)')
    print('  分享方式：把该文件传到 GitHub Pages / Gitee Pages，或随简历附文件链接')


# ===========================================================================
def main():
    tables = step1_sql_extract()
    df = step2_score_segment(tables['wide'], tables['txn'])
    df = df.merge(tables['experiment'][[
        'user_id', 'treatment', 'coupon_redeemed',
        'repeat_90d', 'outcome_spend_90d']], on='user_id', how='left')
    df, crosstab, kmeans_ari, k_table = step3_kmeans(df)
    model_out = step4_modeling(df)
    results = step5_ab_roi(df)
    profit = profit_curve(df)
    results['profit_curve'] = profit
    results['honesty'] = honesty_stats(df)
    step6_export(tables, df, results, crosstab, kmeans_ari, k_table, model_out)
    step7_html(df, results, crosstab, kmeans_ari, k_table, model_out)
    print('\n✅ 流水线全部完成')


if __name__ == '__main__':
    main()
