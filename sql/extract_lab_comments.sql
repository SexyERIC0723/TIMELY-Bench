 -- extract_lab_comments.sql
 -- 提取实验室检测的comment字段（包含异常值描述）

 WITH icu_cohort AS (
     SELECT DISTINCT
         ie.subject_id,
         ie.hadm_id,
         ie.stay_id,
         ie.intime AS icu_intime
     FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
     WHERE ie.los >= 1
 ),

 -- 关键实验室项目（与patterns相关）
 key_lab_items AS (
     SELECT itemid, label, fluid, category
     FROM `physionet-data.mimiciv_3_1_hosp.d_labitems`
     WHERE LOWER(label) LIKE '%creatinine%'
        OR LOWER(label) LIKE '%potassium%'
        OR LOWER(label) LIKE '%lactate%'
        OR LOWER(label) LIKE '%bilirubin%'
        OR LOWER(label) LIKE '%platelet%'
        OR LOWER(label) LIKE '%wbc%'
        OR LOWER(label) LIKE '%white blood%'
        OR LOWER(label) LIKE '%hemoglobin%'
        OR LOWER(label) LIKE '%bicarbonate%'
        OR LOWER(label) LIKE '%ph%'
 ),

 lab_with_comments AS (
     SELECT
         le.subject_id,
         le.hadm_id,
         le.charttime,
         le.itemid,
         di.label AS lab_name,
         le.value,
         le.valuenum,
         le.valueuom,
         le.flag,  -- 'abnormal' 标志
         le.ref_range_lower,
         le.ref_range_upper,
         le.comments AS lab_comment
     FROM `physionet-data.mimiciv_3_1_hosp.labevents` le
     INNER JOIN key_lab_items di ON le.itemid = di.itemid
     WHERE le.comments IS NOT NULL
       AND LENGTH(le.comments) > 5
 )

 SELECT
     c.stay_id,
     c.subject_id,
     c.hadm_id,
     lc.charttime,
     TIMESTAMP_DIFF(lc.charttime, c.icu_intime, HOUR) AS hour_offset,
     lc.lab_name,
     lc.value,
     lc.valuenum,
     lc.valueuom,
     lc.flag,
     lc.ref_range_lower,
     lc.ref_range_upper,
     lc.lab_comment
 FROM icu_cohort c
 INNER JOIN lab_with_comments lc
     ON c.subject_id = lc.subject_id
    AND c.hadm_id = lc.hadm_id
 WHERE TIMESTAMP_DIFF(lc.charttime, c.icu_intime, HOUR) BETWEEN 0 AND 24
 ORDER BY c.stay_id, lc.charttime;