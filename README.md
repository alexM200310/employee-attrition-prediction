# Employee Attrition Prediction & Retention Analysis
## Overview
This project uses employee-level HR data to analyze attrition patterns and develop machine-learning models that identify employees at elevated risk of leaving. The goal is to support proactive retention planning by combining exploratory analysis, predictive modeling, and business recommendations.

## Problem
Employee turnover is costly to replace and disruptive to teams. This project identifies which employee attributes most strongly predict attrition and quantifies how well those attributes can flag at-risk employees ahead of time.

## Dataset
IBM HR Analytics Employee Attrition & Performance (Kaggle), ~1,470 employee records, 35 features including department, overtime status, job satisfaction, tenure, and compensation.

## Approach
- Conducted exploratory data analysis of attrition patterns by department, overtime status, job satisfaction, and tenure
- Removed constant and identifier columns, including `EmployeeCount`, `EmployeeNumber`, `Over18`, and `StandardHours`
- One-hot encoded categorical variables and converted the target variable to binary form: `Yes = 1`, `No = 0`
- Created an 80/20 stratified train-test split to preserve the attrition-class distribution
- Standardized features for logistic regression
- Compared two classification models:
  - Logistic regression baseline using `class_weight="balanced"`
  - Random forest classifier with 300 trees and `class_weight="balanced"`
- Evaluated models using precision, recall, F1-score, confusion matrices, ROC curves, and ROC-AUC instead of relying on accuracy alone

## Results
Logistic regression outperformed the random forest for identifying employee attrition risk, achieving a higher ROC-AUC of 0.80 compared with 0.78. It correctly identified 29 of 47 employees who left, producing an attrition recall of approximately 62%, while the random forest identified only 15 of 47 leavers, or approximately 32% recall. Although logistic regression generated more false positives, it is the preferred model for this retention use case because it captures substantially more at-risk employees and gives HR more opportunities to intervene before turnover occurs.

## Key exploratory findings
Sales had the highest attrition rate across the three departments at approximately 21%, while Research & Development retained about 86% of its employees, the highest retention rate among the departments. Overtime also showed a strong association with attrition: employees who worked overtime had an attrition rate of approximately 31%, compared with about 10% for employees who did not work overtime. Although the no-overtime group was larger overall, the overtime group had slightly more employees leave, which indicates a disproportionately higher risk of attrition among employees working overtime. Finally, employees with the lowest job-satisfaction rating had the highest attrition rate, at approximately 23%, suggesting that lower job satisfaction is associated with a greater risk of departure.

## Recommendation
Both logistic regression and random forest identified overtime and job satisfaction as important predictors of employee attrition. HR should prioritize a review of overtime practices and conduct targeted retention check-ins with employees who regularly work overtime, especially those reporting lower job satisfaction or who are early in their tenure. The exploratory analysis showed that employees working overtime had an attrition rate of approximately 31%, compared with about 10% for employees who did not work overtime, making workload and work-life balance important areas for further investigation. The predictive model can help HR prioritize retention conversations, but it should support—not replace—manager judgment and employee feedback. Because this dataset represents one organization and one historical period, the findings should be validated with the company’s own workforce data before being applied broadly.

## Limitations
Single-company, single-year dataset — drivers may not generalize across industries or time periods. Model is a decision-support signal, not a standalone termination or intervention trigger.

## Stack
Python, Pandas, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook
