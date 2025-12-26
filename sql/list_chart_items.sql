 -- list_chart_items.sql
 -- 列出chartevents中所有可能包含文本评估的项目

 SELECT
     itemid,
     label,
     abbreviation,
     category,
     unitname,
     linksto
 FROM `physionet-data.mimiciv_3_1_icu.d_items`
 WHERE category IN (
     'Routine Vital Signs',
     'Neurological',
     'Respiratory',
     'Cardiovascular',
     'Skin - Impairment',
     'Pain - Loss of Function'
 )
    OR LOWER(label) LIKE '%assessment%'
    OR LOWER(label) LIKE '%note%'
    OR LOWER(label) LIKE '%comment%'
    OR LOWER(label) LIKE '%status%'
 ORDER BY category, label;