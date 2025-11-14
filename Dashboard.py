import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
import os

@st.cache_data
def load_data():
    """
    Loads all available and expected CSV files from the ROOT directory.
    (Sesuai dengan struktur GitHub Anda)
    """
    
    # --- Path ini mengasumsikan file CSV ada di ROOT (level yang sama dengan Dashboard.py) ---
    
    data_files = {
        # --- Core Employee & Dim Data ---
        "employees": "Study Case DA - employees.csv",
        "dim_directorates": "Study Case DA - dim_directorates.csv",
        "tv_tgv_mapping": "Study Case DA - Talent Variable (TV) & Talent Group Variable (TGV).csv",
        "dim_divisions": "Study Case DA - dim_divisions.csv",
        "dim_departments": "Study Case DA - dim_departments.csv",
        "dim_positions": "Study Case DA - dim_positions.csv",
        "dim_areas": "Study Case DA - dim_areas.csv",
        "dim_companies": "Study Case DA - dim_companies.csv",
        "dim_competency_pillars": "Study Case DA - dim_competency_pillars.csv",

        # --- Analysis Data ---
        "performance_ratings": "Study Case DA - performance_yearly.csv",
        "competencies_yearly": "Study Case DA - competencies_yearly.csv",
        "papi_scores": "Study Case DA - papi_scores.csv",
        "profiles_psych": "Study Case DA - profiles_psych.csv",
        "strengths": "Study Case DA - strengths.csv",

        # --- Optional Dim Files ---
        "dim_grades": "Study Case DA - dim_grades.csv",
        "dim_education": "Study Case DA - dim_education.csv"
    }

    df_dict = {}
    missing_files = []

    for key, filename in data_files.items():
        try:
            # Cek apakah file ada
            if os.path.exists(filename):
                df_dict[key] = pd.read_csv(filename)
            else:
                missing_files.append(filename)
                df_dict[key] = None
        except Exception as e:
            st.error(f"Error loading {filename}: {e}")
            df_dict[key] = None

    if missing_files:
        st.error(f"File not found: {', '.join(missing_files)}. Please make sure all files are available in your GitHub repository (di folder utama).")

    # --- Post-processing after load ---
    if df_dict.get("strengths") is not None:
        if 'theme' in df_dict["strengths"].columns:
            df_dict["strengths"] = df_dict["strengths"].rename(columns={'theme': 'strength_name'})

    return df_dict, missing_files


def clean_data(df):
    """
    Applies mean imputation for numeric and mode for non-numeric missing values.
    """
    if df is None:
        return None
        
    df_clean = df.copy()
    
    numeric_cols = df_clean.select_dtypes(include=np.number).columns
    non_numeric_cols = df_clean.select_dtypes(exclude=np.number).columns
    
    if not numeric_cols.empty:
        if all(df_clean[col].isnull().all() for col in numeric_cols):
            pass
        else:
            try:
                num_imputer = SimpleImputer(strategy='mean')
                df_clean[numeric_cols] = num_imputer.fit_transform(df_clean[numeric_cols])
            except ValueError as e:
                st.warning(f"Could not impute numeric data: {e}")
                
    if not non_numeric_cols.empty:
        if all(df_clean[col].isnull().all() for col in non_numeric_cols):
            pass
        else:
            try:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df_clean[non_numeric_cols] = cat_imputer.fit_transform(df_clean[non_numeric_cols])
            except ValueError as e:
                st.warning(f"Could not impute non-numeric data: {e}")
                
    return df_clean


def main():
    st.set_page_config(layout="wide")
    st.title("Step 1: Discover the Pattern of Success")
    st.markdown("Exploring what differentiates high-performing (Rating 5) employees from others.")

    df_dict, missing_files = load_data()
    
    # ✅ FIX: Ganti pesan error untuk membaca dari root
    if any(df is None for df in [df_dict.get('employees'), df_dict.get('performance_ratings')]):
        st.error(r"Core files 'Study Case DA - employees.csv' or 'Study Case DA - performance_yearly.csv' are missing. The app cannot run.")
        return

    # --- Process Performance Data ---
    employees_df = df_dict.get('employees')
    perf_ratings_df = df_dict.get('performance_ratings')
    
    # Clean the base employees dataframe
    employees_clean = clean_data(employees_df)
    
    # Create a merged analysis dataframe
    analysis_df = employees_clean.copy()

    # Clean performance ratings before use
    perf_ratings_df = clean_data(perf_ratings_df)
    
    # Process real performance data
    # Use the latest rating for each employee
    latest_ratings = perf_ratings_df.sort_values('year', ascending=False).drop_duplicates('employee_id')
    
    # Handle outliers (as per brief) - cap ratings at 5
    latest_ratings['rating'] = latest_ratings['rating'].apply(lambda x: min(x, 5) if pd.notnull(x) else x)
    
    latest_ratings['high_performer'] = latest_ratings['rating'] == 5 # As per brief, 5 is high performer
    
    # Merge with employees_df
    analysis_df = pd.merge(analysis_df, latest_ratings[['employee_id', 'rating', 'high_performer']], on='employee_id', how='left')
    
    # For analysis, we only care about those WITH ratings.
    analysis_df.dropna(subset=['rating'], inplace=True)
    analysis_df['high_performer'] = analysis_df['high_performer'].fillna(False)

    # --- Create Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview & Contextual", 
        "🧭 Competency Pillars", 
        "🧠 Psychometric Profiles", 
        "💪 Behavioral (Strengths)", 
        "💡 Success Formula Synthesis"
    ])

    # --- Tab 1: Contextual Factors ---
    with tab1:
        st.header("Contextual Factors Analysis")
        st.write("Comparing high performers vs. others based on their roles and background.")

        # Merge with dim tables for labels
        if df_dict.get('dim_positions') is not None:
            analysis_df = pd.merge(analysis_df, df_dict.get('dim_positions'), on='position_id', how='left', suffixes=('', '_pos'))
        
        if df_dict.get('dim_grades') is not None:
            analysis_df = pd.merge(analysis_df, df_dict.get('dim_grades'), on='grade_id', how='left', suffixes=('', '_grade'))
        
        if df_dict.get('dim_education') is not None:
            analysis_df = pd.merge(analysis_df, df_dict.get('dim_education'), on='education_id', how='left', suffixes=('', '_edu'))

        col1, col2 = st.columns(2)
        
        with col1:
            # Viz 1: Years of Service
            st.subheader("Years of Service Distribution")
            fig_yos = px.histogram(analysis_df, x='years_of_service_months', color='high_performer', 
                                   barmode='overlay', marginal='box', title="Distribution of Years of Service")
            st.plotly_chart(fig_yos, use_container_width=True)

            # Viz 2: Grade Level
            st.subheader("Performance by Grade Level")
            # Use 'name_grade' if available, else fallback to 'grade_id'
            grade_col = 'name_grade' if 'name_grade' in analysis_df.columns else 'grade_id'
            
            grade_viz = analysis_df.groupby(grade_col)['high_performer'].mean().reset_index().sort_values(by='high_performer', ascending=False)
            fig_grade = px.bar(grade_viz, x=grade_col, y='high_performer', 
                               title='Proportion of High Performers by Grade', labels={grade_col: 'Grade', 'high_performer': 'High Performer %'})
            st.plotly_chart(fig_grade, use_container_width=True)

        with col2:
            # Viz 3: Position
            st.subheader("Performance by Position")
            # Use 'name_pos' if available, else fallback to 'position_id'
            pos_col = 'name' if 'name' in analysis_df.columns else 'position_id'

            pos_viz = analysis_df.groupby(pos_col)['high_performer'].mean().reset_index().sort_values(by='high_performer', ascending=False)
            fig_pos = px.bar(pos_viz.head(15), x=pos_col, y='high_performer', 
                             title='Top 15 Positions by High Performer %', labels={pos_col: 'Position', 'high_performer': 'High Performer %'})
            st.plotly_chart(fig_pos, use_container_width=True)

            # Viz 4: Education Level
            st.subheader("Performance by Education")
            # Use 'name_edu' if available, else fallback to 'education_id'
            edu_col = 'name_edu' if 'name_edu' in analysis_df.columns else 'education_id'

            edu_viz = analysis_df.groupby(edu_col)['high_performer'].mean().reset_index().sort_values(by='high_performer', ascending=False)
            fig_edu = px.bar(edu_viz, x=edu_col, y='high_performer', 
                             title='Proportion of High Performers by Education', labels={edu_col: 'Education', 'high_performer': 'High Performer %'})
            st.plotly_chart(fig_edu, use_container_width=True)

    # --- Tab 2: Competency Pillars ---
    with tab2:
        st.header("Competency Pillar Analysis")
        comp_df = df_dict.get('competencies_yearly')
        dim_comp = df_dict.get('dim_competency_pillars')
        
        if comp_df is None or dim_comp is None:
            st.warning("`Study Case DA - competencies_yearly.csv` or `dim_competency_pillars.csv` not found.")
        else:
            # --- Start Competency Analysis ---
            
            # 1. Get latest ratings from our main analysis_df
            latest_ratings_comp = analysis_df[['employee_id', 'rating', 'high_performer']]
            
            # 2. Get latest competencies
            latest_comp = clean_data(comp_df).sort_values('year', ascending=False).drop_duplicates(['employee_id', 'pillar_code'])
            
            # 3. Merge
            comp_analysis_df = pd.merge(latest_comp, latest_ratings_comp, on='employee_id', how='inner')
            comp_analysis_df = pd.merge(comp_analysis_df, dim_comp, on='pillar_code', how='left')

            # 4. Viz 1: Radar Chart
            st.subheader("Radar Chart: High Performers vs. Others")
            radar_data = comp_analysis_df.groupby(['high_performer', 'pillar_label'])['score'].mean().reset_index()
            
            # Perbaikan: Cek jika radar_data kosong
            if radar_data.empty:
                st.warning("No competency data available to display radar chart.")
            else:
                radar_pivot = radar_data.pivot(index='high_performer', columns='pillar_label', values='score').reset_index()
                
                # Get categories from dim_comp to ensure order
                categories = dim_comp['pillar_label'].unique().tolist()
                
                # Ensure all categories are present
                for cat in categories:
                    if cat not in radar_pivot.columns:
                        radar_pivot[cat] = np.nan
                
                # Re-order pivot table columns to match categories list
                radar_pivot = radar_pivot[['high_performer'] + categories]

                fig_radar = go.Figure()
                
                # High Performer Trace
                try:
                    hp_data_row = radar_pivot[radar_pivot['high_performer'] == True]
                    if not hp_data_row.empty:
                        hp_data = hp_data_row.iloc[0]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=hp_data[categories].values, 
                            theta=categories, 
                            fill='toself', 
                            name='High Performers (Rating 5)'
                        ))
                except Exception as e:
                    st.warning(f"Could not draw High Performer radar: {e}")
                
                # Others Trace
                try:
                    op_data_row = radar_pivot[radar_pivot['high_performer'] == False]
                    if not op_data_row.empty:
                        op_data = op_data_row.iloc[0]
                        fig_radar.add_trace(go.Scatterpolar(
                            r=op_data[categories].values, 
                            theta=categories, 
                            fill='toself', 
                            name='Others (Rating < 5)'
                        ))
                except Exception as e:
                    st.warning(f"Could not draw Others radar: {e}")
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
                    showlegend=True,
                    title="Competency Scores: High Performers vs. Others"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # 5. Viz 2: Heatmap (Correlation)
            st.subheader("Competency Correlation Heatmap")
            comp_pivot = comp_analysis_df.pivot_table(index='employee_id', columns='pillar_label', values='score')
            comp_pivot_merged = pd.merge(comp_pivot, latest_ratings_comp[['employee_id', 'rating']], on='employee_id')
            
            # Calculate correlation
            corr = comp_pivot_merged.corr(numeric_only=True)
            
            # Filter to show correlation with 'rating' only
            if 'rating' in corr:
                corr_rating = corr[['rating']].sort_values(by='rating', ascending=False)
                corr_rating = corr_rating.drop('rating', errors='ignore') # Drop self-correlation
                
                fig_heatmap = px.imshow(corr_rating, text_auto=True, 
                                        title="Correlation of Competencies with Overall Rating",
                                        color_continuous_scale='RdBu_r', range_color=[-1, 1],
                                        labels={'index': 'Competency Pillar'})
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.write("This chart shows which competencies have the strongest positive (blue) or negative (red) relationship with the performance `rating`.")
            else:
                st.warning("Could not calculate competency correlation.")

    # --- Tab 3: Psychometric Profiles ---
    with tab3:
        st.header("Psychometric Profile Analysis")
        st.write("Analysis of `papi_scores.csv` and `profiles_psych.csv`.")
        
        if df_dict.get('papi_scores') is None or df_dict.get('profiles_psych') is None:
            st.warning("`Study Case DA - papi_scores.csv` or `Study Case DA - profiles_psych.csv` not found.")
        else:
            # --- Start Psychometric Analysis ---
            psych_df = clean_data(df_dict.get('profiles_psych'))
            
            # PAPI needs to be pivoted
            papi_raw = clean_data(df_dict.get('papi_scores'))
            papi_df = papi_raw.pivot_table(index='employee_id', columns='scale_code', values='score').reset_index()
            
            # Merge with performance
            latest_ratings_psych = analysis_df[['employee_id', 'rating', 'high_performer']]
            psych_analysis_df = pd.merge(psych_df, latest_ratings_psych, on='employee_id', how='inner')
            papi_analysis_df = pd.merge(papi_df, latest_ratings_psych, on='employee_id', how='inner')

            col1, col2 = st.columns(2)

            with col1:
                # Viz 1: DISC
                st.subheader("Performance by DISC Profile")
                if 'disc' in psych_analysis_df.columns:
                    disc_viz = psych_analysis_df.groupby('disc')['high_performer'].mean().reset_index().sort_values(by='high_performer', ascending=False)
                    fig_disc = px.bar(disc_viz, x='disc', y='high_performer', title='Proportion of High Performers by DISC Type',
                                      labels={'disc': 'DISC Type', 'high_performer': 'High Performer %'})
                    st.plotly_chart(fig_disc, use_container_width=True)
                else:
                    st.warning("`disc` column not found in `profiles_psych.csv`")

                # Viz 2: MBTI
                st.subheader("Performance by MBTI Profile")
                if 'mbti' in psych_analysis_df.columns:
                    mbti_viz = psych_analysis_df.groupby('mbti')['high_performer'].mean().reset_index().sort_values(by='high_performer', ascending=False)
                    fig_mbti = px.bar(mbti_viz, x='mbti', y='high_performer', title='Proportion of High Performers by MBTI Type',
                                      labels={'mbti': 'MBTI Type', 'high_performer': 'High Performer %'})
                    st.plotly_chart(fig_mbti, use_container_width=True)
                else:
                    st.warning("`mbti` column not found in `profiles_psych.csv`")
            
            with col2:
                # Viz 3: PAPI Correlation
                st.subheader("PAPI Factor Correlation with Rating")
                if not papi_analysis_df.empty:
                    papi_corr_cols = papi_analysis_df.columns.drop(['employee_id', 'rating', 'high_performer'], errors='ignore')
                    
                    # Ensure columns are numeric before calculating correlation
                    for col in papi_corr_cols:
                        papi_analysis_df[col] = pd.to_numeric(papi_analysis_df[col], errors='coerce')
                    
                    papi_corr = papi_analysis_df.corr(numeric_only=True)[['rating']].sort_values(by='rating', ascending=False)
                    papi_corr = papi_corr.drop(['rating', 'high_performer'], errors='ignore') # Drop self-correlation
                    
                    fig_papi_corr = px.imshow(papi_corr, text_auto=True, 
                                            title="Correlation of PAPI Factors with Overall Rating",
                                            color_continuous_scale='RdBu_r', range_color=[-1, 1],
                                            labels={'index': 'PAPI Factor'})
                    fig_papi_corr.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_papi_corr, use_container_width=True)
                else:
                    st.warning("Could not generate PAPI correlation chart.")

    # --- Tab 4: Behavioral (Strengths) ---
    with tab4:
        st.header("Behavioral (Strengths) Analysis")
        st.write("Analysis of `strengths.csv`. (Note: 'theme' column is used as `strength_name`)")
        
        if df_dict.get('strengths') is None:
            st.warning(r"`Study Case DA - strengths.csv` not found.")
        else:
            # --- Start Strengths Analysis ---
            strengths_df = clean_data(df_dict.get('strengths')) # Already renamed 'theme' to 'strength_name' in load_data
            tv_mapping_df = df_dict.get('tv_tgv_mapping')
            
            # Merge with performance
            latest_ratings_strength = analysis_df[['employee_id', 'rating', 'high_performer']]
            strengths_analysis_df = pd.merge(strengths_df, latest_ratings_strength, on='employee_id', how='inner')

            # Viz 1: Top Strengths Comparison
            st.subheader("Top 10 Strengths: High Performers vs. Others")
            
            col1, col2 = st.columns(2)
            
            with col1:
                hp_strengths = strengths_analysis_df[strengths_analysis_df['high_performer'] == True]['strength_name'].value_counts().nlargest(10).reset_index()
                fig_hp_str = px.bar(hp_strengths, x='strength_name', y='count', title="Top 10 Strengths (High Performers)",
                                    labels={'strength_name': 'Strength', 'count': 'Count'},
                                    color_discrete_sequence=['#1f77b4']) # Blue
                st.plotly_chart(fig_hp_str, use_container_width=True)

            with col2:
                op_strengths = strengths_analysis_df[strengths_analysis_df['high_performer'] == False]['strength_name'].value_counts().nlargest(10).reset_index()
                fig_op_str = px.bar(op_strengths, x='strength_name', y='count', title="Top 10 Strengths (Others)",
                                    labels={'strength_name': 'Strength', 'count': 'Count'},
                                    color_discrete_sequence=['#ff7f0e']) # Orange
                st.plotly_chart(fig_op_str, use_container_width=True)

            # Viz 2: TGV Analysis
            st.subheader("Strength Analysis by Talent Group Variable (TGV)")
            if tv_mapping_df is not None:
                # Map strengths to TGVs
                strength_to_tgv = tv_mapping_df.dropna(subset=['Talent Group Variable (TGV)'])
                # Filter for CliftonStrengths
                strength_to_tgv = strength_to_tgv[strength_to_tgv['Test as Talent Variable (TV)'] == 'CliftonStrengths']
                strength_to_tgv = strength_to_tgv[['Sub-test', 'Talent Group Variable (TGV)']].rename(columns={'Sub-test': 'strength_name'})
                
                strengths_with_tgv = pd.merge(strengths_analysis_df, strength_to_tgv, on='strength_name', how='left')
                strengths_with_tgv['Talent Group Variable (TGV)'] = strengths_with_tgv['Talent Group Variable (TGV)'].fillna('Other')

                # Count TGVs per person
                tgv_counts = strengths_with_tgv.groupby(['employee_id', 'high_performer', 'Talent Group Variable (TGV)']).size().reset_index(name='tgv_count')
                
                # Average counts by performance
                tgv_viz_data = tgv_counts.groupby(['high_performer', 'Talent Group Variable (TGV)'])['tgv_count'].mean().reset_index()
                
                fig_tgv = px.bar(tgv_viz_data, x='Talent Group Variable (TGV)', y='tgv_count', color='high_performer', 
                                 barmode='group', title="Average TGV-mapped Strengths per Person",
                                 labels={'Talent Group Variable (TGV)': 'Talent Group', 'tgv_count': 'Avg. Strengths per Person'})
                st.plotly_chart(fig_tgv, use_container_width=True)
            else:
                st.warning("`Study Case DA - Talent Variable (TV) & Talent Group Variable (TGV).csv` is needed for TGV analysis.")

    # --- Tab 5: Success Formula Synthesis ---
    with tab5:
        st.header("Success Formula (Draft)")
        st.write("This synthesis is a **hypothetical example** based on the analyses in the other tabs. Once you have analyzed the real data from the charts, this formula must be updated.")
        
        st.subheader("How to use this tab")
        st.markdown("""
        1.  Go through Tabs 1-4 and identify the key differentiators.
        2.  **Look for strong signals:**
            * **Competencies:** Which pillars are *much* higher for HPs in the Radar Chart? Which have high correlation (blue bars) in the heatmap?
            * **Psychometrics:** Do HPs cluster in certain DISC or MBTI types? Which PAPI factors have strong correlations (positive or negative)?
            * **Behavioral:** What are the Top 3 strengths for HPs that *don't* appear in the Top 3 for "Others"?
            * **Contextual:** Are HPs concentrated in specific Grades or Positions?
        3.  Update the "Key Findings" and "Proposed Success Formula" below with your **real findings**.
        """)
        
        st.subheader("Hypothetical Key Findings (REPLACE THIS)")
        st.markdown("""
        * **Competencies:** High performers (HPs) score significantly higher in 'Commercial Savvy & Impact' (CSI) and 'Lead, Inspire & Empower' (LIE). 'Quality Delivery Discipline' (QDD) is a high baseline for *all* employees, but doesn't differentiate HPs.
        * **Contextual:** Employees in 'Grade 5+' are 3x more likely to be HPs. 'Years of Service' shows a U-shape; HPs are often 2-4 years (high potential) or 10+ years (deep experts).
        * **Psychometrics:** A 'Dominance' (D) DISC profile and 'Analytical' (PAPI 'A' score > 7) show a strong positive correlation with a 'Rating 5'.
        * **Behavioral:** The 'Strategic' and 'Achiever' (CliftonStrengths) are present in >50% of HPs, vs. <15% for others.
        """)
        
        st.subheader("Proposed Success Formula (Rule-Based Framework)")
        st.markdown("""
        This formula uses a combination of 'Must-Have' (gates) and 'Differentiator' (weighted) factors, grouped by TGV.
        
        ---
        
        #### **Gate 1: Foundation (Must-Have)**
        *If these are not met, the candidate is a poor fit regardless of other scores.*
        
        * (Competency) `Quality Delivery Discipline` (QDD) Score >= 4
        * (Context) `Grade` is 'Grade 3' or higher
        
        ---
        
        #### **TGV 1: Leadership & Impact (Weight: 40%)**
        *Measures ability to drive commercial outcomes and lead others.*
        
        * **TV 1.1 (Comp):** `Lead, Inspire & Empower` (LIE) Score >= 4  *(Weight: 50%)*
        * **TV 1.2 (Comp):** `Commercial Savvy & Impact` (CSI) Score >= 4  *(Weight: 30%)*
        * **TV 1.3 (Psych):** `DISC` Profile is 'D' or 'i'  *(Weight: 20%)*
        
        *TGV 1 Score = (TV 1.1 * 0.5) + (TV 1.2 * 0.3) + (TV 1.3 * 0.2)*
        
        ---
        
        #### **TGV 2: Cognitive & Strategic Agility (Weight: 35%)**
        *Measures problem-solving and-thinking capabilities.*
        
        * **TV 2.1 (Strength):** Has 'Strategic' or 'Analytical' Strength  *(Weight: 40%)*
        * **TV 2.2 (Psych):** `PAPI 'A'` (Analytical) Score >= 7  *(Weight: 30%)*
        * **TV 2.3 (Comp):** `Forward Thinking & Clarity` (FTC) Score >= 3  *(Weight: 30%)*
        
        *TGV 2 Score = (TV 2.1 * 0.4) + (TV 2.2 * 0.3) + (TV 2.3 * 0.3)*
        
        ---
        
        #### **TGV 3: Drive & Experience (Weight: 25%)**
        *Measures resilience and proven application of skills.*
        
        * **TV 3.1 (Strength):** Has 'Achiever' Strength  *(Weight: 40%)*
        * **TV 3.2 (Comp):** `Growth Drive & Resilience` (GDR) Score >= 4  *(Weight: 30%)*
        * **TV 3.3 (Context):** `years_of_service_months` is 24-48 OR > 120  *(Weight: 30%)*
        
        *TGV 3 Score = (TV 3.1 * 0.4) + (TV 3.2 * 0.3) + (TV 3.3 * 0.3)*
        
        ---
        
        ### **Final Success Score Calculation**
        
        `Final Score = IF(Gate_1_Met, (TGV1_Score * 0.40) + (TGV2_Score * 0.35) + (TGV3_Score * 0.25), 0)`
        
        *(Note: All TV scores would be normalized (e.g., 0 or 1 for binary flags, 0-1 scale for numeric scores) before weighting.)*
        """)

if __name__ == "__main__":
    main()
