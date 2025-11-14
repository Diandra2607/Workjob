import os
import streamlit as st
import pandas as pd
import sqlalchemy
from sqlalchemy import text
import requests
import json
import numpy as np
import plotly.express as px

# --- Configuration (direct Supabase URL) ---
# Replace xxxxx with actual project id.
DATABASE_URL = "postgresql://postgres:0cKRGuXARrkpzgNz@db.wolwbhqjdrxavtohdydt.supabase.co:5432/postgres?sslmode=require"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "gpt-4o-mini"

# --- DB connection ---
@st.cache_resource
def get_engine():
    return sqlalchemy.create_engine(
        DATABASE_URL,
        connect_args={"sslmode": "require"},
        pool_pre_ping=True
    )

engine = get_engine()

# --- Helpers: OpenRouter LLM call ---
def generate_job_profile_openrouter(role_name, job_level, role_purpose, example_requirements=None):
    system_prompt = (
        "You are an expert talent/HR analyst. Given role metadata, produce: "
        "1) a concise Job Description (2-4 sentences), "
        "2) a bulleted list of Job Requirements / Key Competencies (8-12 bullets), "
        "3) short 'Why this role needs these competencies' summary (2-3 bullets)."
    )
    user_prompt = f"""
Role name: {role_name}
Job level: {job_level}
Role purpose: {role_purpose}

If available, suggested requirements: {example_requirements}

Output JSON with keys: description, requirements (array), competencies_summary (array).
"""
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 700,
        "temperature": 0.2
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    url = "https://api.openrouter.ai/v1/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    text_out = None
    if "choices" in data and len(data["choices"]) > 0:
        text_out = data["choices"][0]["message"]["content"]
    else:
        text_out = data.get("result") or json.dumps(data)

    try:
        parsed = json.loads(text_out)
    except Exception:
        parsed = {
            "description": text_out,
            "requirements": [],
            "competencies_summary": []
        }
    return parsed

# --- Helpers: Insert job vacancy + mapping (records) ---
def create_job_vacancy(conn, role_name, job_level, role_purpose, benchmark_employee_ids):
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS job_vacancies (
        job_vacancy_id SERIAL PRIMARY KEY,
        role_name TEXT,
        job_level TEXT,
        role_purpose TEXT,
        created_at TIMESTAMP DEFAULT now()
    );
    """))
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS job_vacancy_benchmarks (
        id SERIAL PRIMARY KEY,
        job_vacancy_id INTEGER REFERENCES job_vacancies(job_vacancy_id),
        benchmark_employee_id INTEGER
    );
    """))

    result = conn.execute(
        text("INSERT INTO job_vacancies (role_name, job_level, role_purpose) VALUES (:r, :l, :p) RETURNING job_vacancy_id"),
        {"r": role_name, "l": job_level, "p": role_purpose}
    )
    job_vacancy_id = result.scalar_one()

    mappings = [{"job_vacancy_id": job_vacancy_id, "benchmark_employee_id": int(e)} for e in benchmark_employee_ids]
    if len(mappings) > 0:
        conn.execute(
            text("INSERT INTO job_vacancy_benchmarks (job_vacancy_id, benchmark_employee_id) VALUES (:job_vacancy_id, :benchmark_employee_id)"),
            mappings
        )

    return job_vacancy_id

# --- The parameterized matching SQL ---
# This SQL is the core logic you provided, slightly adapted to accept a dynamic 'benchmark list'
# For brevity we embed as a formatted string and use WITH params. We will pass benchmark IDs via a temporary table.
MATCHING_SQL = """
-- We will use a temp table benchmarks_tmp( employee_id ) which we fill prior to running.
WITH BenchmarkData AS (
    SELECT
        tbs.employee_id,
        tbs.avg_competency_score AS avg_competency_score_tv,
        tbs.psych_pauli AS pauli,
        tbs.psych_faxtor AS faxtor,
        tbs.psych_iq AS iq,
        tbs.psych_gtq AS gtq,
        tbs.psych_tiki AS tiki,
        tbs.papi_score AS papi_score_tv,
        tbs.directorate,
        tbs.role,
        tbs.grade
    FROM talent_benchmarks tbs
    INNER JOIN benchmarks_tmp bt ON tbs.employee_id = bt.employee_id
),
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
        s.theme AS strength_theme
    FROM talent_benchmarks tbs
    LEFT JOIN strengths s ON tbs.employee_id = s.employee_id AND s.rank = 1
),
TV_TGV_Mapping AS (
    SELECT 'avg_competency_score_tv' AS tv_name, 'Core Competencies' AS tgv_name, 1 AS higher_is_better UNION ALL
    SELECT 'pauli', 'Motivation & Drive', 1 UNION ALL
    SELECT 'faxtor', 'Social Orientation & Collaboration', 1 UNION ALL
    SELECT 'iq', 'Cognitive Complexity & Problem-Solving', 1 UNION ALL
    SELECT 'gtq', 'Cognitive Complexity & Problem-Solving', 1 UNION ALL
    SELECT 'tiki', 'Cognitive Complexity & Problem-Solving', 1 UNION ALL
    SELECT 'papi_score_tv', 'PAPI General Score', 1 UNION ALL
    SELECT 'strength_theme', 'Top Strength Theme', 0
),
NumericBaselines AS (
    SELECT 'avg_competency_score_tv' AS tv_name, CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_competency_score_tv) AS TEXT) AS baseline_score FROM BenchmarkData UNION ALL
    SELECT 'pauli', CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pauli) AS TEXT) FROM BenchmarkData UNION ALL
    SELECT 'faxtor', CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY faxtor) AS TEXT) FROM BenchmarkData UNION ALL
    SELECT 'iq', CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY iq) AS TEXT) FROM BenchmarkData UNION ALL
    SELECT 'gtq', CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gtq) AS TEXT) FROM BenchmarkData UNION ALL
    SELECT 'tiki', CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tiki) AS TEXT) FROM BenchmarkData UNION ALL
    SELECT 'papi_score_tv', CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY papi_score_tv) AS TEXT) FROM BenchmarkData
),
StrengthMode AS (
    SELECT
        s.theme AS baseline_score
    FROM strengths s
    WHERE s.rank = 1 AND s.employee_id IN (SELECT employee_id FROM BenchmarkData)
    GROUP BY s.theme
    ORDER BY COUNT(*) DESC
    LIMIT 1
),
BenchmarkBaselines AS (
    SELECT * FROM NumericBaselines
    UNION ALL
    SELECT 'strength_theme' AS tv_name, baseline_score FROM StrengthMode
),
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
    FROM AllEmployeeTVs aet
    CROSS JOIN TV_TGV_Mapping t
    INNER JOIN BenchmarkBaselines b ON t.tv_name = b.tv_name
),
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
        CASE
            WHEN tv_name = 'strength_theme' THEN
                CASE WHEN user_score = baseline_score THEN 100.0 ELSE 0.0 END
            WHEN higher_is_better = 1 THEN
                (CAST(user_score AS REAL) / CAST(baseline_score AS REAL)) * 100.0
            WHEN higher_is_better = 0 THEN
                -- Assuming 0 means 'closer is better', but logic for 'lower is better' or 'mode match' might be needed
                -- For now, this is a placeholder. The original logic for 'strength_theme' handles non-numeric
                -- Let's refine the 'strength_theme' logic from original SQL.
                -- The original SQL implies non-numeric match only for strength_theme
                -- The original SQL had 'higher_is_better=0' for strength_theme, let's use the explicit logic
                ((2 * CAST(baseline_score AS REAL) - CAST(user_score AS REAL)) / CAST(baseline_score AS REAL)) * 100.0
            ELSE 0.0
        END AS tv_match_rate_raw
    FROM UnpivotedTVs
    -- Need to handle potential division by zero if baseline_score is 0 for numeric fields
    WHERE (tv_name = 'strength_theme') OR (tv_name <> 'strength_theme' AND CAST(baseline_score AS REAL) <> 0)
),
TGVMatchRate AS (
    SELECT
        employee_id,
        tgv_name,
        AVG(tv_match_rate_raw) AS tgv_match_rate
    FROM TVMatchRate
    GROUP BY employee_id, tgv_name
),
FinalMatchRate AS (
    SELECT
        employee_id,
        AVG(tgv_match_rate) AS final_match_rate
    FROM TGVMatchRate
    GROUP BY employee_id
)
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
FROM TVMatchRate tvr
INNER JOIN TGVMatchRate tgvr ON tvr.employee_id = tgvr.employee_id AND tvr.tgv_name = tgvr.tgv_name
INNER JOIN FinalMatchRate fmr ON tvr.employee_id = fmr.employee_id
ORDER BY tvr.employee_id, tvr.tgv_name, tvr.tv_name;
"""

# --- Run the matching flow ---
def run_matching(conn, benchmark_employee_ids):
    # FIX: Ubah 'integer' menjadi 'text' agar sesuai dengan skema talent_benchmarks
    conn.execute(text("CREATE TEMP TABLE IF NOT EXISTS benchmarks_tmp (employee_id text) ON COMMIT DROP;"))
    conn.execute(text("TRUNCATE TABLE benchmarks_tmp;"))
    if benchmark_employee_ids:
        # FIX: Masukkan ID sebagai string (text), bukan integer
        rows = [{"employee_id": e} for e in benchmark_employee_ids] 
        conn.execute(text("INSERT INTO benchmarks_tmp (employee_id) VALUES (:employee_id)"), rows)
    df = pd.read_sql(text(MATCHING_SQL), conn)
    return df

# --- Streamlit UI ---
st.set_page_config(page_title="AI Talent Match Dashboard", layout="wide")
st.title("AI Talent Match & Job Profile Builder")

with st.sidebar.form("inputs"):
    st.header("Job Vacancy Inputs")
    role_name = st.text_input("Role Name", value="Data Analyst")
    job_level = st.selectbox("Job Level", ["Junior", "Middle", "Senior", "Lead"], index=1)
    role_purpose = st.text_area("Role Purpose (1-2 sentences)", value="Analyze data to inform product and business decisions.")
    benchmark_ids_text = st.text_input("Selected benchmark employee IDs (comma-separated)", value="312,335,175")
    example_requirements = st.text_area("(Optional) Example Requirements / Hints for LLM", value="")
    submitted = st.form_submit_button("Run Analysis")

if submitted:
    benchmark_employee_ids = [x.strip() for x in benchmark_ids_text.split(",") if x.strip()] # Kita izinkan non-digit dulu

    if len(benchmark_employee_ids) == 0:
        st.error("Please provide at least one benchmark employee ID.")
    else:
        st.info(f"Using benchmarks: {', '.join(benchmark_employee_ids)}")

        with engine.begin() as conn:
            job_vacancy_id = create_job_vacancy(conn, role_name, job_level, role_purpose, benchmark_employee_ids)
            st.success(f"Created job_vacancy_id = {job_vacancy_id}")
            
            # --- FIX: Pindahkan 'df' DULU, baru cek ---
            df = run_matching(conn, benchmark_employee_ids)

        # --- INI ADALAH PERBAIKAN PENTING ('Cara 2') ---
        if df.empty:
            st.warning("Analysis failed: No data returned from SQL query.")
            st.error("Pastikan Benchmark Employee IDs (contoh: 312, 335) yang Anda masukkan BENAR-BENAR ADA di tabel 'talent_benchmarks' Anda.")
        
        else:
            # --- SEMUA VISUALISASI AMAN DI SINI ---
            final = df[['employee_id', 'final_match_rate']].drop_duplicates().sort_values(['final_match_rate'], ascending=False)

            try:
                with engine.connect() as conn:
                    # Pastikan 'employees' adalah tabel yang benar
                    emp_names = pd.read_sql("SELECT employee_id, full_name FROM employees WHERE employee_id IN :ids", conn, params={"ids": tuple(final.employee_id.unique().astype(str))})
                    final = final.merge(emp_names, on='employee_id', how='left')
            except Exception as e:
                st.warning(f"Could not get employee names (table 'employees' missing?): {e}")
                final['full_name'] = 'N/A' # Beri nilai default

            st.subheader("Ranked Talent List")
            st.dataframe(final.rename(columns={"employee_id": "Employee ID", "full_name": "Name", "final_match_rate": "Final Match Rate (%)"}))

            st.subheader("Per-Employee TGV Overview (Top TGVs & Gaps)")
            tgv_scores = df[['employee_id', 'tgv_name', 'tgv_match_rate']].drop_duplicates()
            top_tgvs = tgv_scores.sort_values(['employee_id', 'tgv_match_rate'], ascending=[True, False]).groupby('employee_id').head(3)
            st.dataframe(top_tgvs)

            st.subheader("Match Rate Distribution")
            fig = px.histogram(final, x="final_match_rate", nbins=20, labels={"final_match_rate": "Final Match Rate (%)"}, title="Distribution of Final Match Rate")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("TGV Heatmap for Top Candidates")
            topN = st.slider("Top N candidates", min_value=3, max_value=20, value=6)
            top_emps = final.head(topN).employee_id.tolist()
            
            # Pastikan top_emps tidak kosong
            if top_emps:
                heat = tgv_scores[tgv_scores.employee_id.isin(top_emps)].pivot(index='employee_id', columns='tgv_name', values='tgv_match_rate').fillna(0)
                st.dataframe(heat)
            else:
                st.info("Not enough candidates to display heatmap.")

            # Ambil best_emp, tapi cek dulu 'final' tidak kosong (meskipun 'if df.empty' sudah menangani ini)
            if not final.empty:
                best_emp = final.iloc[0].employee_id
                st.subheader(f"Profile of Top Candidate: {best_emp}")
                best_tgvs = tgv_scores[tgv_scores.employee_id == best_emp]
                if not best_tgvs.empty:
                    fig2 = px.bar_polar(best_tgvs, r='tgv_match_rate', theta='tgv_name', title=f"Top Candidate {best_emp} - TGV Match Rates", template="plotly")
                    st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("AI-Generated Job Profile (OpenRouter)")
            try:
                job_profile = generate_job_profile_openrouter(role_name, job_level, role_purpose, example_requirements)
                
                # --- FIX: Memperbaiki typo 'st' ---
                st.markdown("**Job Description**")
                st.write(job_profile.get("description", "No description generated."))
                
                st.markdown("**Job Requirements / Key Competencies**")
                requirements = job_profile.get("requirements", [])
                if requirements:
                    for r in requirements:
                        st.write(f"- {r}")
                else:
                    st.write("No requirements generated.")

                st.markdown("**Why these competencies**")
                summary = job_profile.get("competencies_summary", [])
                if summary:
                    for s in summary:
                        st.write(f"- {s}")
                else:
                    st.write("No summary generated.")
                        
            except Exception as e:
                st.error(f"OpenRouter call failed: {e}")
                st.info("You can still use the SQL outputs and visuals above.")
            
            # Pindahkan Tombol Download ke dalam 'else'
            st.download_button("Download full raw results (CSV)", df.to_csv(index=False), file_name=f"matching_results_job_{job_vacancy_id}.csv", mime="text/csv")
