CREATE OR REPLACE TABLE `timely-bench-mimic.timelybench.features_timeseries_24h` AS

WITH co AS (
    SELECT subject_id, hadm_id, stay_id, intime 
    FROM `timely-bench-mimic.timelybench.cohort_base`
),

-- 1. 生成 0-23 的小时骨架
hours AS (
    SELECT h FROM UNNEST(GENERATE_ARRAY(0, 23)) AS h
),
skeleton AS (
    SELECT co.stay_id, co.subject_id, co.hadm_id, co.intime, h as hour
    FROM co CROSS JOIN hours
),

-- 2. 提取生命体征 (Vitals)
vitals AS (
    SELECT 
        co.stay_id,
        TIMESTAMP_DIFF(v.charttime, co.intime, HOUR) AS hour,
        AVG(v.heart_rate) as heart_rate,
        AVG(v.sbp) as sbp,
        AVG(v.dbp) as dbp,
        AVG(v.mbp) as mbp,
        AVG(v.resp_rate) as resp_rate,
        AVG(v.temperature) as temperature,
        AVG(v.spo2) as spo2,
        AVG(v.glucose) as glucose_chart
    FROM co
    JOIN `physionet-data.mimiciv_3_1_derived.vitalsign` v
        ON co.stay_id = v.stay_id
    WHERE v.charttime >= co.intime 
      AND v.charttime < TIMESTAMP_ADD(co.intime, INTERVAL 24 HOUR)
    GROUP BY 1, 2
),

-- 3. 提取化验结果 (Labs) - Chemistry
chem AS (
    SELECT 
        co.stay_id,
        TIMESTAMP_DIFF(le.charttime, co.intime, HOUR) AS hour,
        AVG(le.albumin) as albumin,
        AVG(le.bun) as bun,
        AVG(le.creatinine) as creatinine,
        AVG(le.glucose) as glucose_lab,
        AVG(le.sodium) as sodium,
        AVG(le.potassium) as potassium,
        AVG(le.bicarbonate) as bicarbonate,
        AVG(le.chloride) as chloride,
        AVG(le.aniongap) as aniongap
    FROM co
    JOIN `physionet-data.mimiciv_3_1_derived.chemistry` le
        ON co.hadm_id = le.hadm_id 
    WHERE le.charttime >= co.intime 
      AND le.charttime < TIMESTAMP_ADD(co.intime, INTERVAL 24 HOUR)
    GROUP BY 1, 2
),

-- 4. 提取血常规 (CBC)
cbc AS (
    SELECT 
        co.stay_id,
        TIMESTAMP_DIFF(le.charttime, co.intime, HOUR) AS hour,
        AVG(le.wbc) as wbc,
        AVG(le.hemoglobin) as hemoglobin,
        AVG(le.hematocrit) as hematocrit,
        AVG(le.platelet) as platelet
    FROM co
    JOIN `physionet-data.mimiciv_3_1_derived.complete_blood_count` le
        ON co.hadm_id = le.hadm_id 
    WHERE le.charttime >= co.intime 
      AND le.charttime < TIMESTAMP_ADD(co.intime, INTERVAL 24 HOUR)
    GROUP BY 1, 2
),

-- 5. 提取 GCS
gcs_data AS (
    SELECT 
        co.stay_id,
        TIMESTAMP_DIFF(g.charttime, co.intime, HOUR) AS hour,
        MIN(g.gcs) as gcs_min 
    FROM co
    JOIN `physionet-data.mimiciv_3_1_derived.gcs` g
        ON co.stay_id = g.stay_id
    WHERE g.charttime >= co.intime 
      AND g.charttime < TIMESTAMP_ADD(co.intime, INTERVAL 24 HOUR)
    GROUP BY 1, 2
),

-- 6. 提取尿量 (Urine Output)
uo_data AS (
    SELECT 
        co.stay_id,
        TIMESTAMP_DIFF(uo.charttime, co.intime, HOUR) AS hour,
        SUM(uo.urineoutput) as urineoutput
    FROM co
    JOIN `physionet-data.mimiciv_3_1_derived.urine_output` uo
        ON co.stay_id = uo.stay_id
    WHERE uo.charttime >= co.intime 
      AND uo.charttime < TIMESTAMP_ADD(co.intime, INTERVAL 24 HOUR)
    GROUP BY 1, 2
), -- <-- 修复：这里必须加逗号

-- 7. 增加血气分析 (BG)
bg_data AS (
    SELECT 
        co.stay_id,
        -- 使用 TIMESTAMP_DIFF 保持一致性
        TIMESTAMP_DIFF(bg.charttime, co.intime, HOUR) AS hour,
        AVG(bg.lactate) as lactate,
        AVG(bg.ph) as ph
    FROM co
    JOIN `physionet-data.mimiciv_3_1_derived.bg` bg 
        ON co.hadm_id = bg.hadm_id
    WHERE bg.charttime >= co.intime 
      AND bg.charttime < TIMESTAMP_ADD(co.intime, INTERVAL 24 HOUR)
    GROUP BY 1, 2
)

-- 8. 汇总
SELECT 
    s.stay_id,
    s.hour,
    -- Vitals
    v.heart_rate, v.sbp, v.dbp, v.mbp, v.resp_rate, v.temperature, v.spo2, v.glucose_chart,
    -- Labs
    c.albumin, c.bun, c.creatinine, c.glucose_lab, c.sodium, c.potassium, c.bicarbonate, c.chloride, c.aniongap,
    -- CBC
    cbc.wbc, cbc.hemoglobin, cbc.hematocrit, cbc.platelet,
    -- Blood Gas (修复：增加这两个字段)
    bg.lactate, bg.ph,
    -- Other
    g.gcs_min,
    u.urineoutput
FROM skeleton s
LEFT JOIN vitals v ON s.stay_id = v.stay_id AND s.hour = v.hour
LEFT JOIN chem c ON s.stay_id = c.stay_id AND s.hour = c.hour
LEFT JOIN cbc cbc ON s.stay_id = cbc.stay_id AND s.hour = cbc.hour
LEFT JOIN gcs_data g ON s.stay_id = g.stay_id AND s.hour = g.hour
LEFT JOIN uo_data u ON s.stay_id = u.stay_id AND s.hour = u.hour
LEFT JOIN bg_data bg ON s.stay_id = bg.stay_id AND s.hour = bg.hour -- <-- 修复：增加 JOIN 逻辑
ORDER BY s.stay_id, s.hour;