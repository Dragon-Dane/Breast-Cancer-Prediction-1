# Breast Cancer Prediction
Predict the status of breast cancer using Wisconsin Breast Cancer dataset
Background information of the data and its source: http://mlr.cs.umass.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Breast cancer is the most common cancer among women and one of the major causes of death among women worldwide. Every year approximately 124 out of 100,000 women are diagnosed with breast cancer, and the estimation is that 23 out of the 124 women will die of this disease.

This dataset was created by Dr. William, Nick Street and Olvi in University of Wisconsin. The dataset contains 569 rows and 32 columns and it is widely used for this kind of application because it has a pretty large number of records and there are only a few missing values. 

The data set contains 32 total variables, many of which we will use to predict the eventual malignant or benign category of each tumor suspected of cancer. For our project, we will investigate this clinical data set to examine how each variable relates to a patient’s diagnosis, and present our findings on how to better predict a patient’s diagnosis based on these variables. Our target variable is ‘Diagnosis’, which can take on one of two possible values, ‘Malignant’ (M), or Benign (B). The Predictor variables in our dataset are various measures of the patient’s tumor that is suspected of being cancerous.

Our goal for this project is to build different classification models using R and Python (pandas, numpy, sklearn, matplotlib) and choose the best model based on various model evaluation matrices.  The data set we will be using contains a wide range of information relating to each patient’s tumor that is being tested, which we will then use to predict the likelihood of a malignant vs. benign diagnosis.

# RESOURCES

- The detail of the process you can find in the project submission pdf file.
- The dataset is named data.csv
- The R code file is named Data_Preprocessing.R for data preparation steps
- The Python code file is named Cancer_Project_Script.py for model building

# FINAL MODEL

The final model that we have chosen for the issue in hand is “Support Vector Machines” for the following reasons:
•	High Accuracy, Precision, Recall and AUC-ROC numbers. The statistics of the chosen model are better than all the other models that we have built. Since we want to catch as many cancerous tumors as possible, we require a high recall which is given by the model (0.94).  
•	Since SVM is used when we have a good number of features, it well suits our data set.
•	Also, we have less number of observations. All the other models require good amount of data to train and test, whereas SVM performs well even when the number of observations are limited.
