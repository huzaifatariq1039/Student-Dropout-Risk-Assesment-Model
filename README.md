# Student-Dropout-Risk-Assesment-Model
📌 Overview
This project is an end-to-end machine learning system for predicting and analyzing student dropout risks using the UCI Predict Students Dropout and Academic Success dataset.

It not only predicts dropout probability but also:

Generates comprehensive data visualizations

Performs feature engineering

Compares multiple ML models

Produces a detailed intervention plan

Saves all plots and reports automatically

🚀 Key Features
1. Data Loading & Preprocessing
Handles different CSV separators (;, ,, \t)

Standardizes column names with a mapping for the UCI dataset

Fills missing values:

Numerical → median

Categorical → mode

Verifies Target column existence

2. Exploratory Data Analysis (EDA)
Creates high-quality plots (300 DPI) for:

Target Distribution (bar + pie charts)

Age Distribution by outcome

Academic Performance (box plots for grades)

Financial Factors (tuition status, debtor, scholarship)

Parent Education Impact on dropout rates

3. Feature Engineering
Adds 11 new features, such as:

Academic performance indicators

Financial stress metrics

Family support indicators

Age-based risk flags

4. Model Training & Evaluation
Trains and compares:

Random Forest

Gradient Boosting

Logistic Regression

Support Vector Machine (SVM)

Evaluates each model with:

Accuracy

AUC Score

ROC Curves

Confusion Matrices (raw & normalized)

Automatically selects the best model by AUC score.

5. Feature Importance
Ranks top 20 features by importance for tree-based models

Saves visualization as 07_feature_importance.png

6. Risk Analysis
Categorizes students into:

Critical Risk (≥70%)

High Risk (50–69%)

Medium Risk (30–49%)

Low Risk (<30%)

Generates:

Risk level distribution chart

Dropout probability histogram

Actual dropout rate per risk level

High-risk student characteristics breakdown

7. Intervention Recommendations
Priority Matrix for risk groups

Strategy Effectiveness chart

Cost-Benefit Analysis

Resource Allocation Pie Chart

Generates intervention_report.txt with:

Summary statistics

Recommended interventions

Estimated budgets

KPIs

Implementation timeline

8. Single Student Predictions
Predicts dropout probability, risk category, and outcome for individual students

Accepts dictionary or DataFrame input

📊 Generated Files
All plots are stored in dropout_analysis_plots/:

File	Description
01_target_distribution.png	Target distribution analysis
02_academic_performance_analysis.png	Academic performance by outcome
03_financial_factors_analysis.png	Financial factors impact
04_parent_education_impact.png	Parent education impact
05_model_performance_comparison.png	Accuracy & AUC comparison + ROC curves
06_confusion_matrix.png	Confusion matrices (raw & normalized)
07_feature_importance.png	Top 20 important features
08_risk_analysis.png	Risk level distribution & dropout probability
09_high_risk_student_characteristics.png	Profile of high-risk students
10_intervention_recommendations.png	Intervention strategies, costs, allocations
intervention_report.txt	Detailed intervention plan

📂 Project Structure
bash
Copy
Edit
.
├── main.py                # Main project script
├── data/                  # Place your dataset here
├── dropout_analysis_plots/ # Auto-generated visualizations & reports
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
🛠 Installation
1️⃣ Clone Repository

bash
Copy
Edit
git clone https://github.com/your-username/student-dropout-risk-assessment.git
cd student-dropout-risk-assessment
2️⃣ Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
📌 Usage
Download Dataset
Get the dataset from UCI and save it as:

kotlin
Copy
Edit
data/data.csv
Run the Script

bash
Copy
Edit
python main.py
Check Outputs
See generated PNG and TXT files in dropout_analysis_plots/.

📈 Example Output
Risk Level Distribution

📜 Requirements
Python 3.8+

pandas, numpy, matplotlib, seaborn, scikit-learn

📜 License
This project is released under the MIT License.

