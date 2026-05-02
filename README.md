# People Pulse – Employee Attrition Prediction & Data Mining Analysis

Repository: EmployeeAttrition-DataMiningAnalysis

## Project Overview
This project analyzes and predicts employee attrition using a dataset of 10,000 employee records with 26 features.

The analysis focuses on identifying high-risk employee groups and key factors influencing turnover, such as income, overtime, tenure, work-life balance, and manager relationship, to support data-driven HR retention strategies.

## Dataset Overview
- 10,000 employee records
- 26 features
- Target variable: Attrition
- Attrition rate: approximately 18%
- Data type: Structured HR dataset

## Business Objective
The objective of this project is to identify key drivers of employee attrition and evaluate factors influencing turnover to help HR teams shift from reactive hiring to proactive retention strategies.

## Key Challenges
The dataset and business problem presented several analytical challenges:

- Severe class imbalance, with approximately 18% attrition cases
- Multiple interacting HR factors influencing turnover
- Weak separation between employee behavior patterns
- Need to identify both predictive performance and interpretable business insights
- Translating model results into actionable HR recommendations

## Workflow
1. Data Understanding
2. Exploratory Data Analysis
3. Data Quality Check
   - Accuracy
   - Completeness
   - Consistency
   - Timeliness
   - Uniqueness
4. Data Preparation
   - Encoding
   - Feature Selection
   - SMOTE Oversampling
   - Standardization
5. Classification Analysis in RapidMiner
   - Logistic Regression
   - Random Forest
   - Neural Network
6. Clustering Analysis in RapidMiner
   - K-Means Clustering
7. Association Rule Mining with Python
8. Business Insight and Recommendation

## Tools & Technologies
- Python: pandas, matplotlib, seaborn
- Scikit-learn
- imbalanced-learn / SMOTE
- mlxtend
- RapidMiner
- Association Rule Mining
- Data Mining
- Data Visualization

## Data Preparation Summary
| Process | Description |
|---|---|
| Encoding | Converted categorical variables using label encoding and one-hot encoding |
| Feature Selection | Selected relevant features using correlation analysis and Random Forest feature importance |
| SMOTE | Balanced the attrition class to improve high-risk employee detection |
| Standardization | Scaled numerical features for model readiness |
| CSV Export | Exported prepared datasets for further analysis in RapidMiner |

## Model Performance
| Model | Accuracy | Precision (Attrition) | Recall (Attrition) |
|---|---:|---:|---:|
| Logistic Regression | 72.81% | 71.73% | 75.28% |
| Random Forest | 82.51% | 87.45% | 75.91% |
| Neural Network | 77.31% | 77.17% | 77.56% |

Random Forest achieved the best overall performance, especially in identifying high-risk employees with strong precision.

## Advanced Analysis
Beyond predictive modeling, K-Means clustering and Association Rule Mining were applied to uncover deeper behavioral patterns.

Association Rule Mining revealed that combinations of low income, poor manager relationships, non-managerial roles, and specific department patterns were linked to higher attrition risk.

Example rule:

- Low income + poor manager relationship → higher attrition risk
- Confidence: 81%
- Lift: 1.65

## Key Insights
- Overtime is strongly linked to higher attrition, highlighting workload imbalance.
- Lower income is a key driver of employee turnover.
- Employees with shorter tenure show higher attrition risk.
- Poor work-life balance significantly contributes to attrition.
- Attrition is driven by multiple interacting factors, not a single cause.

## Business Recommendations
- Prioritize retention strategies for high-risk employees.
- Adjust compensation structures for lower-income employee segments.
- Improve workload and overtime management.
- Strengthen onboarding and engagement programs for early-stage employees.
- Use employee segmentation to support targeted HR interventions.

## Final Report
The full project report is available in the `report/` folder.

## Project Structure

```text
employee-attrition-prediction-data-mining/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   └── Pre.py
│
├── outputs/
│   ├── scaled_resampled.csv
│   └── df_binned_only.csv
│
├── report/
│   └── employee_attrition_report.pdf
│
└── images/
    ├── attrition_distribution.png
    ├── smote_before_after.png
    ├── feature_importance.png
    ├── model_performance.png
    └── association_rules.png
