"""Streamlit app - Student Performance Early-Warning System.

Run locally:  streamlit run app/streamlit_app.py
Deployed from the same GitHub repository on Streamlit Community Cloud.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent


def find(name, required=True):
    """Locate a project file wherever it sits in the repository.
    Works for the normal layout (app/, src/, models/) and also if the files were uploaded flat."""
    for base in (HERE, HERE.parent):
        hit = next(base.rglob(name), None)
        if hit is not None:
            return hit
    if required:
        raise FileNotFoundError(f"{name} was not found in the repository - check that src/, models/ and docs/ were uploaded.")
    return None


sys.path.insert(0, str(find("common.py").parent))
from common import AT_RISK, CLASS_ORDER, add_features  # noqa: E402  (same code as training)

st.set_page_config(page_title="Student Early-Warning System", page_icon="🎓", layout="wide")


@st.cache_resource
def load_model():
    model = joblib.load(find("student_performance_model.joblib"))
    meta = json.load(open(find("model_metadata.json")))
    return model, meta


model, meta = load_model()
ENGINEERED_OUT = [c for c in meta["features"] if c in {
    "avg_assessment_score", "midterm_vs_coursework", "online_engagement", "study_per_credit", "total_weekly_load",
    "attendance_x_submission", "academic_gap", "low_connectivity", "n_missing_values"}]
RAW_FEATURES = [c for c in meta["features"] if c not in ENGINEERED_OUT]
RISK_COLORS = {"Low": "#3C9D6B", "Medium": "#E0A526", "High": "#D1495B"}


def score(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the SAME feature engineering as training, then predict."""
    full = add_features(raw_df)
    proba = model.predict_proba(full[meta["features"]])
    classes = list(model.named_steps["model"].classes_)
    out = pd.DataFrame(proba, columns=classes, index=raw_df.index)
    out["risk_score"] = out[AT_RISK].sum(axis=1)
    out["predicted_status"] = [classes[i] for i in proba.argmax(axis=1)]
    out["risk_level"] = pd.cut(out["risk_score"], [-0.01, 0.35, 0.60, 1.01], labels=["Low", "Medium", "High"])
    return out


def actions(row, level):
    tips = []
    if row.get("attendance_rate", 100) < 65: tips.append("Attendance is low: contact the student and agree an attendance plan.")
    if row.get("lms_logins_week", 99) < 5: tips.append("Very low LMS activity: send a nudge and check access to course materials.")
    if row.get("assignment_submission_rate", 100) < 65: tips.append("Many assignments not submitted: offer a deadline-recovery meeting.")
    if row.get("late_submissions", 0) >= 6: tips.append("Frequent late submissions: refer to time-management support.")
    if row.get("internet_access") in ("Limited", "Unreliable") or row.get("device_access") in ("Phone Only", "Shared Device"):
        tips.append("Connectivity/device barrier: offer campus laptop loan or lab access.")
    if row.get("employment_hours_week", 0) >= 25: tips.append("High work hours: discuss flexible deadlines or a reduced load.")
    if not tips and level == "Low": tips.append("No action needed. Keep monitoring each term.")
    if not tips: tips.append("Book an advising meeting to discuss workload and study habits.")
    return tips


st.title("🎓 Student Performance Early-Warning System")
st.caption("Predicts a student's end-of-term outcome (Fail / Withdrawn / Pass / Distinction) and a risk score so advisors "
           "can support the right students early. Decision-support tool, not an automatic decision-maker.")

tab_single, tab_batch, tab_model = st.tabs(["Single student", "Batch scoring (CSV)", "About the model"])

# ----------------------------------------------------------------- single student
with tab_single:
    st.subheader("Enter student information")
    values = {}
    groups = {
        "Profile": ["academic_year", "semester", "institution_type", "program", "study_level", "year_of_study", "age",
                    "enrollment_status", "admission_type", "course_difficulty", "course_load_credits"],
        "Background": ["prior_gpa", "entry_exam_score", "scholarship_status", "financial_aid", "commute_minutes",
                       "employment_hours_week", "internet_access", "device_access"],
        "Behaviour this term": ["attendance_rate", "absence_days", "study_hours_week", "library_visits_month", "lms_logins_week",
                                "video_completion_rate", "forum_posts", "support_sessions"],
        "Assessment so far": ["assignment_submission_rate", "assignment_avg_score", "quiz_avg_score", "midterm_score", "late_submissions"],
    }
    cols = st.columns(4)
    for col, (g, names) in zip(cols, groups.items()):
        with col:
            st.markdown(f"**{g}**")
            for c in names:
                if c not in RAW_FEATURES:
                    continue
                label = c.replace("_", " ").capitalize()
                if c in meta["categorical"]:
                    opts = meta["choices"][c]
                    values[c] = st.selectbox(label, opts, index=opts.index(meta["modes"][c]))
                else:
                    lo, hi = meta["ranges"][c]
                    if c in ("age",): lo, hi = 16.0, 60.0
                    if c.endswith("_rate") or c.endswith("_score"): lo, hi = 0.0, 100.0
                    if c == "prior_gpa": lo, hi = 0.0, 4.0
                    default = float(min(max(meta["medians"][c], lo), hi))
                    is_int = c in ("year_of_study", "age", "absence_days", "library_visits_month", "forum_posts",
                                   "late_submissions", "course_load_credits", "lms_logins_week", "support_sessions")
                    if is_int:
                        values[c] = st.number_input(label, int(lo), int(hi), int(round(default)), step=1)
                    else:
                        values[c] = st.number_input(label, float(lo), float(hi), round(default, 1), step=0.5)

    if st.button("Predict outcome", type="primary"):
        row = pd.DataFrame([values])
        res = score(row).iloc[0]
        level = str(res["risk_level"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted outcome", res["predicted_status"])
        c2.metric("At-risk probability (Fail + Withdrawn)", f"{res['risk_score'] * 100:.1f}%")
        c3.markdown(f"<div style='padding:14px;border-radius:8px;background:{RISK_COLORS[level]};color:white;text-align:center;"
                    f"font-size:22px'>Risk level: {level}</div>", unsafe_allow_html=True)
        st.markdown("**Probability of each outcome**")
        st.bar_chart(pd.DataFrame({"probability": [res[c] for c in CLASS_ORDER]}, index=CLASS_ORDER))
        st.markdown("**Suggested advisor actions**")
        for t in actions(values, level):
            st.write("• " + t)
        if res["predicted_status"] in ("Pass", "Distinction") and level != "Low":
            st.info("The most likely outcome is a pass, but the risk score is not low: monitor this student.")

# ----------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Score a whole cohort")
    st.write("Upload a CSV with the same raw columns as the dataset (the `performance_status` column is not needed). "
             "Missing values are handled automatically.")
    sample_path = find("student_performance_sample_5000.csv", required=False)
    if sample_path is not None:
        st.download_button("Download a sample file", sample_path.read_bytes(), "student_sample.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in RAW_FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            data = data.head(50000)
            res = score(data[RAW_FEATURES])
            out = pd.concat([data.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
            st.write(f"Scored {len(out):,} students.")
            st.bar_chart(out["risk_level"].value_counts().reindex(["Low", "Medium", "High"]))
            st.dataframe(out.sort_values("risk_score", ascending=False).head(200))
            st.download_button("Download scored file", out.to_csv(index=False).encode(), "scored_students.csv", "text/csv")

# ----------------------------------------------------------------- about
with tab_model:
    st.subheader("About the model")
    st.write(f"**Algorithm:** {meta['model_name']}  |  **Test accuracy:** {meta['test_accuracy']:.1%}  |  "
             f"**Test macro-F1:** {meta['test_macro_f1']:.3f}  |  **At-risk AUC:** {meta['at_risk_auc']:.3f}")
    st.write("Business purpose: help advisors find students likely to **fail or withdraw** early enough to intervene. "
             "Risk levels: Low < 35%, Medium 35–60%, High > 60% at-risk probability.")
    imp_path = find("feature_importance.csv", required=False)
    if imp_path is not None:
        imp = pd.read_csv(imp_path).head(12).set_index("feature")["importance"]
        st.markdown("**What drives predictions (permutation importance)**")
        st.bar_chart(imp)
    st.warning("Limitations: trained on a synthetic dataset; predictions show statistical association, not causes; "
               "check for bias across groups before any real-world use; always keep a human in the loop.")
