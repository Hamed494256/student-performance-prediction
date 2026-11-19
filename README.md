# Student Performance Early-Warning System

## 1. Business problem
Advisors cannot give every student individual attention. The micro-decision is:
*"Should this student be contacted for extra support this term?"*

We predict each student's end-of-term outcome (**Fail / Withdrawn / Pass / Distinction**) from
information available around mid-term (attendance, assessment scores, assignment behaviour,
background). We also create an **at-risk score** (probability of Fail or Withdrawn) so advisors
can start with the students who need help most.

## 2. Team members and roles
| Member | GitHub username | Role |
|---|---|---|
| H00494256 | [Hamed Almarri] | [e.g. data cleaning and EDA] |
| H00537676 | [Hamed Alblooshi] | [e.g. modelling, tuning and evaluation] |


## 3. Dataset
- **File:** `student_performance_synthetic_raw.csv` (synthetic), 250,800 rows x 34 columns.
- **Target:** `performance_status` (Fail, Withdrawn, Pass, Distinction).
- **Source and licence:** [who generated the data, link or generation code, licence].
- **Approval:** the course instructor approved the use of this synthetic dataset (Milestone 2) on [15/nov/2026].
- **How to obtain:** the raw file is in `data/raw/` (48 MB, below GitHub's 100 MB limit).
  A 5,000-row sample is in `data/sample/`.
- **Quality issues found:** missing values (13 columns, 1-3.6% each), 800 duplicate records,
  13 spellings for 8 programs, 258 impossible values (age, attendance above 100, midterm above 100).
- Data dictionary: `docs/data_dictionary.md`. Dataset search evidence: `docs/dataset_search_evidence.md`.

## 4. Folder structure
```
data/raw/         original CSV (never edited)
data/processed/   clean, train and test files (created by the scripts, not committed)
data/sample/      5,000-row sample
notebooks/        01_... to 07_... Jupyter notebooks (run in order)
src/              the same steps as .py scripts + common.py (shared with the app)
app/              streamlit_app.py
models/           student_performance_model.joblib + model_metadata.json
docs/             data dictionary, search evidence, figures/, results/ (all tables as CSV)
requirements.txt
```

## 5. How to run
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd src
python 01_data_understanding_cleaning.py
python 02_eda_feature_engineering.py
python 03_model_selection.py
python 04_tuning_evaluation.py     # about 10 minutes on one CPU core
python 05_grouped_importance_and_fairness.py
python 06_export_predictions.py
python 07_cost_analysis.py
cd ..
streamlit run app/streamlit_app.py
```
The notebooks in `notebooks/` contain the same code. All random seeds are fixed (`random_state=42`).
`requirements.txt` pins scikit-learn, because the saved model must be loaded with the version that trained it.

## 6. Method summary
1. **Cleaning:** removed 800 duplicates, standardised program labels (13 to 8), set 258 impossible values to missing. Missing values are imputed inside the model pipeline (median / most frequent) to avoid data leakage.
2. **EDA and features:** 6 charts, 9 engineered features (for example average assessment score, online engagement, attendance x submission).
3. **Models:** baseline, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, compared with 3-fold cross-validation (stratified 80/20 split: 200,000 train / 50,000 test).
4. **Tuning:** RandomizedSearchCV (3-fold, macro-F1). The final model was chosen by cross-validated score before looking at the test set.
5. **Evaluation:** one evaluation on the hold-out test set, plus permutation importance, a fairness check across student groups and a cost-based threshold analysis.
6. **Deployment:** Streamlit app with single-student prediction, batch CSV scoring and a model explanation tab.

## 7. Results (50,000 hold-out students, final model = tuned Random Forest)
| Metric | Value |
|---|---|
| Accuracy (4 classes) | 58.4% (majority-class baseline 54.1%) |
| Macro-F1 (4 classes) | 0.629 (baseline 0.176) |
| At-risk recall / precision at threshold 0.5 | 67.7% / 59.1% |
| At-risk ROC-AUC | 0.853 |
| Share of all at-risk students inside the top 20% highest scores | 52.4% (lift 2.6x) |

- The most important information is attendance, assessment scores and assignment submission.
- Pass vs Distinction is the hardest pair to separate.
- At-risk recall differs by less than 5 percentage points across the student groups checked.

## 8. Limitations
- The data are synthetic, so real-world performance will probably be lower.
- A random split was used. The model was not tested on a future academic year.
- Tuning used a 40,000-row sample of the training set (limited computing resources).
- Results show association, not cause.
- The tool supports advisors. It must not be used for automatic decisions about students.

## 9. Live app
[app/streamlit_app.py]
