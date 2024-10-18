Walmart(A retail store )that has multiple outlets USA are facing issues in managing the inventory - to match the demand with respect to supply.
I came up with useful insights using the data and made prediction models to forecast the sales for future months/years.

**Customer Churn Rate Estimator-Duration(Jan 2023-Feb 2023)-**
Objective:Analysing customer data to prevent churn and retain customers at a telecom company ‘Neo’. Utilising data science techniques to derive insights and implement strategies to minimize customer attrition.
Technologies and methodologies-Linear Regression

**The KMeans clustering with the Iris dataset**(Feb 2023-March 2023)
This Project is a classic example of unsupervised learning, where the goal is to group similar data points into clusters without using predefined labels. The Iris dataset is commonly used for this task because it contains easily interpretable data, consisting of measurements for three species of iris flowers.

Here’s a breakdown of how to apply the KMeans algorithm on the Iris dataset in an unsupervised learning project:

1. Iris Dataset Overview
The Iris dataset contains:

150 instances of iris flowers.
4 features: sepal length, sepal width, petal length, and petal width.
3 species: Iris setosa, Iris versicolor, and Iris virginica (though in unsupervised learning, we won't use these labels directly for training).
2. KMeans Clustering
KMeans is a centroid-based clustering algorithm that aims to partition the data into k clusters. Each data point is assigned to the nearest cluster center (centroid). The goal is to minimize the sum of squared distances between data points and their nearest cluster centroid.

**A diabetes classifier project** (March 2023-April 2023)
This is typically aimed at predicting whether an individual has diabetes based on medical and demographic data. These projects utilize machine learning algorithms to classify individuals as diabetic or non-diabetic, based on various features like age, glucose level, body mass index (BMI), and more.

Here’s an overview of what such a project typically involves:

1. Problem Statement
The goal is to build a machine learning model that can accurately predict whether a patient has diabetes based on diagnostic features. This is a binary classification problem where the target variable (Outcome) indicates whether the person has diabetes (1) or not (0).

2. Dataset
One popular dataset used in Kaggle's diabetes classification projects is the Pima Indians Diabetes Dataset. This dataset includes 768 observations of females of Pima Indian heritage aged 21 and older, with 8 medical predictors and 1 binary target variable:

Pregnancies: Number of times pregnant.
Glucose: Plasma glucose concentration.
Blood Pressure: Diastolic blood pressure.
Skin Thickness: Triceps skinfold thickness.
Insulin: 2-Hour serum insulin.
BMI: Body Mass Index.
Diabetes Pedigree Function: A function which scores likelihood of diabetes based on family history.
Age: Age in years.
Outcome: 1 if diabetic, 0 otherwise.
3. Data Preprocessing
Before training a model, data preprocessing steps are crucial, such as:

Handling Missing Values: Some values, like insulin levels, may have missing data, which can be imputed or treated accordingly.
Feature Scaling: Features such as glucose or BMI may require scaling for some machine learning models.
Splitting the Data: The dataset is split into training and testing sets to evaluate model performance.
4. Exploratory Data Analysis (EDA)
In the EDA phase, the dataset is analyzed to find relationships between features. For example:

Check the distribution of glucose and BMI levels for diabetic and non-diabetic patients.
Investigate correlation between features like age, pregnancies, and diabetes outcome.
Visualize class distribution to check for imbalance in the target variable.
5. Model Selection
Several machine learning models can be used for this classification task:

Logistic Regression: A simple baseline model often used in binary classification problems.
Random Forest: A more complex model that can capture non-linear relationships in the data.
Support Vector Machines (SVM): A model that can create a decision boundary to classify the outcome.
K-Nearest Neighbors (KNN): A non-parametric method that classifies a new point based on the majority label of its neighbors.
6. Model Evaluation
The models are evaluated using metrics such as:

Accuracy: Measures the overall correctness of the model.
Precision, Recall, and F1 Score: Useful when dealing with class imbalance, these metrics provide a better understanding of false positives and false negatives.
ROC-AUC Curve: Measures the trade-off between sensitivity and specificity.
7. Model Tuning
To improve model performance, techniques like cross-validation and hyperparameter tuning (using GridSearchCV or RandomizedSearchCV) are employed. This helps in finding the best model configuration.

****Supervised Stroke Model(April 2023-May 2023)**
**- This project aims at predicting strokes for patients.According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.

Attribute Information
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
