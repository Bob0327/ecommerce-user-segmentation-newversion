-- ============================================================================
-- pipeline_01_wide_table.sql —— 统一流水线的 SQL 段（取数 + 建宽表）
-- 分工原则（对应项目设计决策 1.B）：
--   SQL 负责：数据审计、筛窗口、按用户聚合成"一人一行"的原始指标宽表
--   Python 负责：百分位打分、规则分层、KMeans 聚类、逻辑回归建模、ROI
-- 后续 Python 通过 pandas.read_sql 直接读 pipeline_wide 表，不走 CSV 中转。
-- ============================================================================

-- @section audit_rowcounts
SELECT 'users' AS 表名, COUNT(*) AS 行数 FROM users
UNION ALL SELECT 'transactions', COUNT(*) FROM transactions
UNION ALL SELECT 'sessions',     COUNT(*) FROM sessions
UNION ALL SELECT 'experiment',   COUNT(*) FROM experiment;

-- @section audit_dates
SELECT 'transactions.purchase_date' AS 字段, MIN(purchase_date) AS 最早, MAX(purchase_date) AS 最晚
FROM transactions
UNION ALL
SELECT 'sessions.session_date', MIN(session_date), MAX(session_date) FROM sessions;

-- @section audit_amounts
SELECT COUNT(*) AS 总订单数, MIN(order_value) AS 最小金额,
       ROUND(AVG(order_value), 2) AS 平均金额, ROUND(MAX(order_value), 2) AS 最大金额,
       SUM(CASE WHEN order_value <= 0 THEN 1 ELSE 0 END) AS 非正金额订单数
FROM transactions;

-- ---------------------------------------------------------------------------
-- @section wide_table
-- 一人一行的原始指标宽表。注意：这里没有打分、没有分层——
-- 那些是"建模决策"，留在 Python 里做，方便调参和评估（决策与取数分离）。
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS pipeline_wide;
CREATE TABLE pipeline_wide AS
WITH
txn AS (
    SELECT * FROM transactions
    WHERE purchase_date >= '2025-01-01' AND purchase_date <= '2025-06-30'
),
txn_agg AS (
    SELECT user_id,
           MAX(purchase_date)  AS last_purchase_date,
           COUNT(*)            AS frequency_180d,
           SUM(order_value)    AS monetary_180d,
           AVG(order_value)    AS avg_order_value_180d
    FROM txn GROUP BY user_id
),
sess AS (
    SELECT * FROM sessions
    WHERE session_date >= '2025-01-01' AND session_date <= '2025-06-30'
),
session_agg AS (
    SELECT user_id, COUNT(*) AS sessions_180d,
           SUM(minutes) AS total_minutes, SUM(pages_viewed) AS total_pages
    FROM sess GROUP BY user_id
),
recent_sess AS (
    SELECT user_id, COUNT(*) AS recent_sessions_30d
    FROM sess
    WHERE session_date > date('2025-06-30', '-30 days')
    GROUP BY user_id
),
base AS (
    SELECT
        u.user_id, u.signup_date, u.age, u.gender, u.location,
        u.income, u.interest, u.newsletter_subscription,
        COALESCE(t.frequency_180d, 0)       AS frequency_180d,
        COALESCE(t.monetary_180d, 0)        AS monetary_180d,
        COALESCE(t.avg_order_value_180d, 0) AS avg_order_value_180d,
        t.last_purchase_date,
        COALESCE(s.sessions_180d, 0)        AS sessions_180d,
        COALESCE(s.total_minutes, 0)        AS total_minutes,
        COALESCE(s.total_pages, 0)          AS total_pages,
        COALESCE(r.recent_sessions_30d, 0)  AS recent_sessions_30d
    FROM users u
    LEFT JOIN txn_agg     t ON u.user_id = t.user_id
    LEFT JOIN session_agg s ON u.user_id = s.user_id
    LEFT JOIN recent_sess r ON u.user_id = r.user_id
)
SELECT
    base.*,
    COALESCE(CAST(julianday('2025-06-30') - julianday(last_purchase_date) AS INTEGER), 999) AS recency_days,
    CAST(total_pages AS REAL) / (frequency_180d + 1) AS friction_index
FROM base;
