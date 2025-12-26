
-- 目标：提取每个病人在进入ICU后前24小时内（0-23小时）的放射科报告元数据及文本
-- 修改点：
-- 1. 强化了 hour_offset 的范围过滤 (0-23)，确保与生理指标骨架表严格对齐
-- 2. 使用 hadm_id 关联，确保医疗逻辑的严密性
-- 3. 增加去重逻辑，确保每个病人每小时最多只有一条记录，防止下游 JOIN 数据爆炸

CREATE OR REPLACE TABLE `timely-bench-mimic.timelybench.note_metadata_24h` AS

WITH rad_notes_raw AS (
    SELECT 
        co.stay_id,
        rad.note_id,
        rad.charttime,
        rad.text AS radiology_text,
        -- 计算相对入室的时间偏移（小时）
        TIMESTAMP_DIFF(rad.charttime, co.intime, HOUR) AS hour_offset
    FROM `timely-bench-mimic.timelybench.cohort_base` co
    JOIN `physionet-data.mimiciv_note.radiology` rad
        ON co.hadm_id = rad.hadm_id -- 使用 hadm_id 关联以确保属于同一次住院
    WHERE rad.charttime >= co.intime 
      -- 初步过滤 24 小时内的数据
      AND rad.charttime < TIMESTAMP_ADD(co.intime, INTERVAL 24 HOUR)
),

filtered_notes AS (
    -- 核心修复：显式强制 hour_offset 落在 0 到 23 之间
    -- 这样可以过滤掉那些刚好在 24 小时 0 分 1 秒产生的边缘数据
    SELECT *
    FROM rad_notes_raw
    WHERE hour_offset >= 0 AND hour_offset < 24
)

SELECT 
    stay_id,
    note_id,
    charttime,
    hour_offset,
    radiology_text
FROM (
    -- 使用窗口函数处理重复：
    -- 如果在同一个小时内（hour_offset 相同）产生了多份报告，
    -- 我们通过 ORDER BY charttime DESC 取该小时内最新的一份，确保序列的唯一性。
    SELECT 
        *,
        ROW_NUMBER() OVER(
            PARTITION BY stay_id, hour_offset 
            ORDER BY charttime DESC
        ) as rn
    FROM filtered_notes
)
WHERE rn = 1 -- 仅保留每小时最新的一份报告
ORDER BY stay_id, hour_offset;