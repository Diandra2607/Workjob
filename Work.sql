UPDATE employees
SET company_id = company_id/ 10
where employee_id is not null;

UPDATE employees
SET area_id = area_id/ 10
where employee_id is not null;

UPDATE employees
SET position_id = position_id/ 10
where employee_id is not null;

UPDATE employees
SET department_id = department_id/ 10
where employee_id is not null;

UPDATE employees
SET division_id = division_id/ 10
where employee_id is not null;

UPDATE employees
SET directorate_id = directorate_id/ 10
where employee_id is not null;

UPDATE employees
SET grade_id = grade_id/ 10
where employee_id is not null;

UPDATE employees
SET education_id = education_id/ 10
where employee_id is not null;

UPDATE employees
SET major_id = major_id/ 10


UPDATE talent_benchmarks
SET company_id =d company_id/ 10
where employee_id is not null;

UPDATE talent_benchmarks
SET area_id = area_id/ 10
where employee_id is not null;

UPDATE talent_benchmarks
SET position_id = position_id/ 10
where employee_id is not null;

UPDATE talent_benchmarks
SET department_id = department_id/ 10
where employee_id is not null;

UPDATE talent_benchmarks
SET division_id = division_id/ 10
where employee_id is not null;

UPDATE talent_benchmarks
SET directorate_id = directorate_id/ 10
where employee_id is not null;

UPDATE talent_benchmarks
SET grade_id = grade_id/ 10
where employee_id is not null;

UPDATE talent_benchmarks
SET education_id = education_id/ 10
where employee_id is not null;

UPDATE talent_benchmarks
SET major_id = major_id/ 10
where employee_id is not null;

ALTER TABLE talent_benchmarks
ADD COLUMN directorate VARCHAR(200);

UPDATE talent_benchmarks
SET directorate = case
WHEN directorate_id = 1 THEN 'Commercial'
WHEN directorate_id = 2 THEN 'HR & Corp Affairs'
ELSE 'Technology'
END;

ALTER TABLE talent_benchmarks
ADD COLUMN role VARCHAR(200);

UPDATE talent_benchmarks
SET role = case
WHEN position_id = 1 THEN 'Brand Executive'
WHEN position_id = 2 THEN 'Data Analyst'
WHEN position_id = 3 THEN 'Finance Officer'
WHEN position_id = 4 THEN 'HRBP'
WHEN position_id = 5 THEN 'Sales Superviser'
ELSE 'Supply Planner'
END;

ALTER TABLE talent_benchmarks
ADD COLUMN grade VARCHAR(200);

UPDATE talent_benchmarks
SET grade = case
WHEN education_id = 1 THEN 'D3'
WHEN education_id = 2 THEN 'S1'
WHEN education_id = 3 THEN 'S2'
ELSE 'SMA'
END;

WITH
-- CTE 1: Define the Benchmark Group and Gather All Talent Data
-- Selects employees with a 'rating' of 5, which defines the ideal benchmark profile.
BenchmarkData AS (
    SELECT
        tbs.employee_id,
        tbs.avg_competency_score AS avg_competency_score_tv, -- Core Competencies TV
        tbs.psych_pauli AS pauli,                             -- Motivation & Drive TV
        tbs.psych_faxtor AS faxtor,                           -- Social Orientation TV
        tbs.psych_iq AS iq,                                   -- Cognitive Complexity TV
        tbs.psych_gtq AS gtq,                                 -- Cognitive Complexity TV
        tbs.psych_tiki AS tiki,                               -- Cognitive Complexity TV
        tbs.papi_score AS papi_score_tv,                      -- PAPI General Score TV
        tbs.directorate,
        tbs.role,
        tbs.grade
    FROM
        talent_benchmarks tbs
    WHERE
        tbs.rating = 5
),
-- CTE 2: Consolidate All Employee Talent Variables for All Employees
-- Combines the TV scores for ALL employees with their rank 1 Strength Theme.
AllEmployeeTVs AS (
    SELECT
        tbs.employee_id,
        tbs.directorate,
        tbs.role,
        tbs.grade,
        tbs.avg_competency_score AS avg_competency_score_tv,
        tbs.psych_pauli AS pauli,
        tbs.psych_faxtor AS faxtor,
        tbs.psych_iq AS iq,
        tbs.psych_gtq AS gtq,
        tbs.psych_tiki AS tiki,
        tbs.papi_score AS papi_score_tv,
        s.theme AS strength_theme -- Employee's Rank 1 Strength
    FROM
        talent_benchmarks tbs
    LEFT JOIN
        strengths s ON tbs.employee_id = s.employee_id AND s.rank = 1
),
-- CTE 3: Map Talent Variables (TV) to Talent Group Variables (TGV) and Define Scoring Direction
-- This acts as a configuration table for the match logic.
TV_TGV_Mapping AS (
    SELECT 'avg_competency_score_tv' AS tv_name, 'Core Competencies' AS tgv_name, 1 AS higher_is_better UNION ALL
    SELECT 'pauli' AS tv_name, 'Motivation & Drive' AS tgv_name, 1 AS higher_is_better UNION ALL
    SELECT 'faxtor' AS tv_name, 'Social Orientation & Collaboration' AS tgv_name, 1 AS higher_is_better UNION ALL
    SELECT 'iq' AS tv_name, 'Cognitive Complexity & Problem-Solving' AS tgv_name, 1 AS higher_is_better UNION ALL
    SELECT 'gtq' AS tv_name, 'Cognitive Complexity & Problem-Solving' AS tgv_name, 1 AS higher_is_better UNION ALL
    SELECT 'tiki' AS tv_name, 'Cognitive Complexity & Problem-Solving' AS tgv_name, 1 AS higher_is_better UNION ALL
    SELECT 'papi_score_tv' AS tv_name, 'PAPI General Score' AS tgv_name, 1 AS higher_is_better UNION ALL
    SELECT 'strength_theme' AS tv_name, 'Top Strength Theme' AS tgv_name, 0 AS higher_is_better -- Categorical, not strictly higher/lower
),
-- CTE 4: Calculate numeric medians + categorical mode separately
NumericBaselines AS (
    SELECT 'avg_competency_score_tv' AS tv_name,
           CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_competency_score_tv) AS TEXT) AS baseline_score
    FROM BenchmarkData
    UNION ALL
    SELECT 'pauli',
           CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pauli) AS TEXT)
    FROM BenchmarkData
    UNION ALL
    SELECT 'faxtor',
           CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY faxtor) AS TEXT)
    FROM BenchmarkData
    UNION ALL
    SELECT 'iq',
           CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY iq) AS TEXT)
    FROM BenchmarkData
    UNION ALL
    SELECT 'gtq',
           CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gtq) AS TEXT)
    FROM BenchmarkData
    UNION ALL
    SELECT 'tiki',
           CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tiki) AS TEXT)
    FROM BenchmarkData
    UNION ALL
    SELECT 'papi_score_tv',
           CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY papi_score_tv) AS TEXT)
    FROM BenchmarkData
),

-- Mode must be computed separately because it requires ORDER BY COUNT
StrengthMode AS (
    SELECT s.theme AS baseline_score
    FROM strengths s
    WHERE s.rank = 1
      AND s.employee_id IN (SELECT employee_id FROM BenchmarkData)
    GROUP BY s.theme
    ORDER BY COUNT(*) DESC
    LIMIT 1
),

BenchmarkBaselines AS (
    SELECT * FROM NumericBaselines
    UNION ALL
    SELECT 'strength_theme' AS tv_name, baseline_score
    FROM StrengthMode
),
-- CTE 5: Unpivot All Employee Talent Variables and Join with Baselines
UnpivotedTVs AS (
    SELECT
        aet.employee_id,
        aet.directorate,
        aet.role,
        aet.grade,
        t.tv_name,
        t.tgv_name,
        b.baseline_score,
        t.higher_is_better,
        -- Select the appropriate user score based on TV type
        CASE
            WHEN t.tv_name = 'avg_competency_score_tv' THEN CAST(aet.avg_competency_score_tv AS TEXT)
            WHEN t.tv_name = 'pauli' THEN CAST(aet.pauli AS TEXT)
            WHEN t.tv_name = 'faxtor' THEN CAST(aet.faxtor AS TEXT)
            WHEN t.tv_name = 'iq' THEN CAST(aet.iq AS TEXT)
            WHEN t.tv_name = 'gtq' THEN CAST(aet.gtq AS TEXT)
            WHEN t.tv_name = 'tiki' THEN CAST(aet.tiki AS TEXT)
            WHEN t.tv_name = 'papi_score_tv' THEN CAST(aet.papi_score_tv AS TEXT)
            WHEN t.tv_name = 'strength_theme' THEN aet.strength_theme
            ELSE NULL
        END AS user_score
    FROM
        AllEmployeeTVs aet
    CROSS JOIN
        TV_TGV_Mapping t
    INNER JOIN
        BenchmarkBaselines b ON t.tv_name = b.tv_name
),
-- CTE 6: Calculate TV Match Rate (Employee x TV)
TVMatchRate AS (
    SELECT
        employee_id,
        directorate,
        role,
        grade,
        tgv_name,
        tv_name,
        baseline_score,
        user_score,
        -- Calculate the Match Rate (Employee x TV)
        CASE
            -- Non-numeric/Categorical (strength_theme)
            WHEN tv_name = 'strength_theme' THEN
                CASE
                    WHEN user_score = baseline_score THEN 100.0
                    ELSE 0.0
                END
            -- Numeric Variables
            WHEN higher_is_better = 1 THEN -- Higher is Better
                (CAST(user_score AS REAL) / CAST(baseline_score AS REAL)) * 100.0
            WHEN higher_is_better = 0 THEN -- Lower is Better (Inverted Ratio)
                -- Formula: ((2 * benchmark_score – user_score) / benchmark_score) * 100.0
                ((2 * CAST(baseline_score AS REAL) - CAST(user_score AS REAL)) / CAST(baseline_score AS REAL)) * 100.0
            ELSE 0.0 -- Default
        END AS tv_match_rate_raw
    FROM
        UnpivotedTVs
),
-- CTE 7: Calculate TGV Match Rate (Employee x TGV)
TGVMatchRate AS (
    SELECT
        employee_id,
        tgv_name,
        -- Use simple average (equal weight) for all TV's within a TGV
        AVG(tv_match_rate_raw) AS tgv_match_rate
    FROM
        TVMatchRate
    GROUP BY
        employee_id, tgv_name
),
-- CTE 8: Calculate Final Match Rate (Employee)
FinalMatchRate AS (
    SELECT
        employee_id,
        -- Use simple average (equal weight) for all TGV's
        AVG(tgv_match_rate) AS final_match_rate
    FROM
        TGVMatchRate
    GROUP BY
        employee_id
)
-- Final Select: Combine all results into the required output format
SELECT
    tvr.employee_id,
    tvr.directorate,
    tvr.role,
    tvr.grade,
    tvr.tgv_name,
    tvr.tv_name,
    CASE 
        WHEN tvr.tv_name = 'strength_theme' THEN '0'
        ELSE ROUND(CAST(NULLIF(tvr.baseline_score, '') AS NUMERIC), 2)::TEXT
    END AS baseline_score,
    CASE 
        WHEN tvr.tv_name = 'strength_theme' THEN tvr.user_score
        ELSE ROUND(CAST(NULLIF(tvr.user_score, '') AS NUMERIC), 2)::TEXT
    END AS user_score,
    ROUND(tvr.tv_match_rate_raw::NUMERIC, 2) AS tv_match_rate,
    ROUND(tgvr.tgv_match_rate::NUMERIC, 2) AS tgv_match_rate,
    ROUND(fmr.final_match_rate::NUMERIC, 2) AS final_match_rate
FROM
    TVMatchRate tvr
INNER JOIN
    TGVMatchRate tgvr
        ON tvr.employee_id = tgvr.employee_id AND tvr.tgv_name = tgvr.tgv_name
INNER JOIN
    FinalMatchRate fmr
        ON tvr.employee_id = fmr.employee_id
ORDER BY
    tvr.employee_id,
    tvr.tgv_name,
    tvr.tv_name;
