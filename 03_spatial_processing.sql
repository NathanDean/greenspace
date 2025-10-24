DROP TABLE IF EXISTS combined_data;

CREATE TABLE combined_data AS
	
	WITH unioned_os AS (
		SELECT ST_Union(geometry) AS geometry
		FROM aligned_os
	),
	
	unioned_ua AS (
		SELECT ST_Union(geometry) AS geometry
		FROM raw_ua_data
	),

	combined_greenspace AS (
		SELECT ST_Union(o.geometry, u.geometry) AS geometry
		FROM unioned_os o, unioned_ua u
	),
	
	exploded_greenspace AS (
		SELECT (ST_Dump(geometry)).geom as geometry
		FROM combined_greenspace
		WHERE geometry IS NOT NULL
	),
	
	lsoa_with_intersecting_greenspace AS (
		SELECT
			l.lsoa,
			l.geometry,
			ST_Intersection(e.geometry, l.geometry) AS greenspace_geometry
		FROM lsoa_with_demographics l
		INNER JOIN exploded_greenspace e
			ON ST_Intersects(l.geometry, e.geometry)
	),
	
	lsoa_with_total_greenspace AS (
		SELECT
			lsoa,
			geometry,
			SUM(ST_Area(greenspace_geometry)) AS total_greenspace_area
		FROM lsoa_with_intersecting_greenspace
		WHERE greenspace_geometry IS NOT NULL
			AND NOT ST_IsEmpty(greenspace_geometry)
		GROUP BY lsoa, geometry
	),
	
	lsoa_with_greenspace_proportions AS (
		SELECT
			lsoa,
			total_greenspace_area / NULLIF(ST_Area(geometry), 0) AS greenspace_proportion
		FROM lsoa_with_total_greenspace
	)

SELECT
	l.*, 
	COALESCE(g.greenspace_proportion, 0) AS greenspace_proportion
FROM lsoa_with_demographics l
LEFT JOIN lsoa_with_greenspace_proportions g USING (lsoa);

ALTER TABLE combined_data
	DROP COLUMN lsoa;