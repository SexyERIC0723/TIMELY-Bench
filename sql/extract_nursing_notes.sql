-- extract_nursing_notes.sql
 -- 从chartevents提取护理评估文本（包含生命体征描述）

 WITH icu_cohort AS (
     SELECT DISTINCT
         ie.subject_id,
         ie.hadm_id,
         ie.stay_id,
         ie.intime AS icu_intime
     FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
     WHERE ie.los >= 1
 ),

 -- 护理评估相关的itemid (需要查d_items确认)
 nursing_items AS (
     SELECT itemid, label, category
     FROM `physionet-data.mimiciv_3_1_icu.d_items`
     WHERE category IN ('Routine Vital Signs', 'Neurological', 'Respiratory', 'Cardiovascular')
        OR label LIKE '%Assessment%'
        OR label LIKE '%Note%'
        OR label LIKE '%Comment%'
 ),

 chart_text_events AS (
     SELECT
         ce.stay_id,
         ce.subject_id,
         ce.charttime,
         ce.itemid,
         di.label AS item_label,
         di.category,
         ce.value AS chart_text,
         ce.valuenum
     FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
     INNER JOIN nursing_items di ON ce.itemid = di.itemid
     WHERE ce.value IS NOT NULL
       AND LENGTH(ce.value) > 10  -- 过滤太短的文本
 )

 SELECT
     c.stay_id,
     c.subject_id,
     c.hadm_id,
     ct.charttime,
     TIMESTAMP_DIFF(ct.charttime, c.icu_intime, HOUR) AS hour_offset,
     ct.item_label,
     ct.category,
     ct.chart_text,
     ct.valuenum
 FROM icu_cohort c
 INNER JOIN chart_text_events ct ON c.stay_id = ct.stay_id
 WHERE TIMESTAMP_DIFF(ct.charttime, c.icu_intime, HOUR) BETWEEN 0 AND 24
 ORDER BY c.stay_id, ct.charttime;
