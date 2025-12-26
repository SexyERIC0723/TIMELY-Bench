 -- extract_discharge_notes.sql
 -- 提取24小时内有出院小结的ICU患者笔记

 WITH icu_cohort AS (
     SELECT DISTINCT
         ie.subject_id,
         ie.hadm_id,
         ie.stay_id,
         ie.intime AS icu_intime,
         ie.outtime AS icu_outtime
     FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
     WHERE ie.los >= 1  -- 至少住ICU 1天
 ),

 discharge_notes AS (
     SELECT
         d.note_id,
         d.subject_id,
         d.hadm_id,
         d.charttime,
         d.text AS discharge_text,
         LENGTH(d.text) AS text_length
     FROM `physionet-data.mimiciv_note.discharge` d
     WHERE d.text IS NOT NULL
       AND LENGTH(d.text) > 10  -- 过滤太短的笔记
 )

 SELECT
     c.stay_id,
     c.subject_id,
     c.hadm_id,
     c.icu_intime,
     d.note_id,
     d.charttime AS note_time,
     TIMESTAMP_DIFF(d.charttime, c.icu_intime, HOUR) AS hour_offset,
     d.discharge_text,
     d.text_length
 FROM icu_cohort c
 INNER JOIN discharge_notes d
     ON c.subject_id = d.subject_id
    AND c.hadm_id = d.hadm_id
 ORDER BY c.stay_id, d.charttime;