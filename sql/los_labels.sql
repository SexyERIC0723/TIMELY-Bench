-- ============================================
-- Step 1: 获取 Prolonged LOS 标签
-- 运行位置: GCP BigQuery
-- ============================================

WITH icu_stays AS (
    SELECT 
        icu.stay_id,
        icu.subject_id,
        icu.hadm_id,
        icu.intime,
        icu.outtime,
        -- 计算ICU住院时长（小时）
        DATETIME_DIFF(icu.outtime, icu.intime, HOUR) as los_hours,
        -- 计算ICU住院时长（天）
        DATETIME_DIFF(icu.outtime, icu.intime, DAY) as los_days
    FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
    WHERE DATETIME_DIFF(icu.outtime, icu.intime, HOUR) >= 24  -- 只保留住院>24h的
),

-- 获取30天再入院信息
readmissions AS (
    SELECT 
        a1.stay_id,
        CASE 
            WHEN MIN(DATETIME_DIFF(a2.intime, a1.outtime, DAY)) <= 30 THEN 1 
            ELSE 0 
        END as readmission_30d
    FROM `physionet-data.mimiciv_3_1_icu.icustays` a1
    LEFT JOIN `physionet-data.mimiciv_3_1_icu.icustays` a2
        ON a1.subject_id = a2.subject_id
        AND a2.intime > a1.outtime
        AND DATETIME_DIFF(a2.intime, a1.outtime, DAY) <= 30
    GROUP BY a1.stay_id
)

SELECT 
    i.stay_id,
    i.subject_id,
    i.hadm_id,
    i.los_hours,
    i.los_days,
    
    -- Prolonged LOS 标签 (多个阈值)
    CASE WHEN i.los_days >= 3 THEN 1 ELSE 0 END as prolonged_los_3d,
    CASE WHEN i.los_days >= 5 THEN 1 ELSE 0 END as prolonged_los_5d,
    CASE WHEN i.los_days >= 7 THEN 1 ELSE 0 END as prolonged_los_7d,
    
    -- 30天再入院标签
    COALESCE(r.readmission_30d, 0) as readmission_30d
    
FROM icu_stays i
LEFT JOIN readmissions r ON i.stay_id = r.stay_id
ORDER BY i.stay_id