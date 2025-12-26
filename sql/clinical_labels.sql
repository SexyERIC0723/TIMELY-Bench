-- 一次性获取所有需要的信息
WITH my_cohort AS (
    -- 你的队列stay_id（从本地cohort.csv上传或手动列出）
    SELECT DISTINCT stay_id
    FROM `timely-bench-mimic.timelybench.cohort_base`  -- 或者直接列出
),

-- 获取ICD诊断
diagnoses AS (
    SELECT 
        icu.stay_id,
        d.icd_code,
        d.icd_version,
        di.long_title,
        d.seq_num
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
    JOIN `physionet-data.mimiciv_3_1_icu.icustays` icu ON d.hadm_id = icu.hadm_id
    LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` di
        ON d.icd_code = di.icd_code AND d.icd_version = di.icd_version
    WHERE icu.stay_id IN (SELECT stay_id FROM my_cohort)
),

-- 获取SOFA（取前24小时的最大值）
sofa_max AS (
    SELECT 
        stay_id,
        MAX(sofa_24hours) as sofa_max,
        MAX(respiration_24hours) as resp_max,
        MAX(cardiovascular_24hours) as cardio_max,
        MAX(renal_24hours) as renal_max
    FROM `physionet-data.mimiciv_3_1_derived.sofa`
    WHERE stay_id IN (SELECT stay_id FROM my_cohort)
      AND hr <= 24
    GROUP BY stay_id
),

-- 获取AKI（取最严重分期）
aki_max AS (
    SELECT 
        stay_id,
        MAX(aki_stage) as aki_stage_max
    FROM `physionet-data.mimiciv_3_1_derived.kdigo_stages`
    WHERE stay_id IN (SELECT stay_id FROM my_cohort)
    GROUP BY stay_id
),

-- 获取Sepsis-3标签
sepsis AS (
    SELECT 
        stay_id,
        sepsis3,
        sofa_score as sepsis_sofa
    FROM `physionet-data.mimiciv_3_1_derived.sepsis3`
    WHERE stay_id IN (SELECT stay_id FROM my_cohort)
)

-- 最终输出
SELECT 
    c.stay_id,
    s.sepsis3,
    s.sepsis_sofa,
    sf.sofa_max,
    a.aki_stage_max,
    -- 聚合该患者的所有ICD码
    STRING_AGG(DISTINCT d.icd_code, ',') as icd_codes,
    STRING_AGG(DISTINCT d.long_title, ' | ') as diagnoses_text
FROM my_cohort c
LEFT JOIN sepsis s ON c.stay_id = s.stay_id
LEFT JOIN sofa_max sf ON c.stay_id = sf.stay_id
LEFT JOIN aki_max a ON c.stay_id = a.stay_id
LEFT JOIN diagnoses d ON c.stay_id = d.stay_id
GROUP BY c.stay_id, s.sepsis3, s.sepsis_sofa, sf.sofa_max, a.aki_stage_max