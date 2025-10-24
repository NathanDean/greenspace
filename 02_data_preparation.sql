CREATE VIEW lsoa_with_demographics AS

-- Align coordinate system with Urban Atlas dataset
WITH aligned_lsoa AS (
    SELECT
        lsoa,
        ST_TRANSFORM(geometry, 3035) AS geometry
    FROM raw_lsoa_data
),

-- Convert very_good_health figures to proportions
health_proportions AS (
    SELECT
        lsoa,
        very_good_health::NUMERIC / NULLIF(all_usual_residents, 0) as very_good_health
    FROM raw_health_data
)

SELECT
    *
FROM aligned_lsoa
INNER JOIN health_proportions USING(lsoa)
INNER JOIN raw_age_sex_data USING(lsoa)
INNER JOIN raw_ethnicity_data USING(lsoa)
INNER JOIN raw_imd_data USING(lsoa);

-- Align coordinate system with Urban Atlas dataset

CREATE VIEW aligned_os AS 
SELECT
    ST_TRANSFORM(geometry, 3035) AS geometry
FROM raw_os_data;