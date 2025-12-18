# Brain-Stroke-Detection

🧠 **Stroke Prediction Using Machine Learning**

This repository contains a complete machine learning pipeline for stroke risk prediction, developed as part of an academic data science project. The study focuses on exploratory data analysis, preprocessing, class imbalance handling, model development, evaluation, and interpretability, using a real-world healthcare dataset.



📌 **Project Overview**

Stroke is one of the leading causes of mortality and long-term disability worldwide. Early identification of individuals at high risk is crucial for timely medical intervention and prevention.

In this project, a binary classification problem is addressed, where the goal is to predict whether a patient is likely to experience a stroke based on demographic, lifestyle, and clinical features. Emphasis is placed on model interpretability and recall, given the high clinical cost of missed stroke cases.



📊 **Dataset**

• Source: Public healthcare dataset (CSV format) on Kaggle

• Target variable: stroke (0 = No stroke, 1 = Stroke)

• Features include:

    • Demographic: age, gender, residence type
    
    • Clinical: hypertension, heart disease, BMI, average glucose level
    
    • Lifestyle: smoking status, work type, marital status



📈 **Key Findings**

• Age, hypertension, heart disease, smoking status, and average glucose level emerged as the most influential predictors. 

• SMOTE successfully balanced the training data but did not significantly improve performance over the baseline model.

• The baseline Logistic Regression model provided stable performance, strong recall for stroke cases, and superior interpretability.

• ROC-AUC values indicated good discriminative ability across validation and test sets.



🎯 **Conclusion**

This project demonstrates how a complete, interpretable, and clinically meaningful machine learning pipeline can be developed for stroke risk prediction. The results highlight the importance of careful preprocessing, appropriate evaluation metrics, and model transparency in medical applications.
