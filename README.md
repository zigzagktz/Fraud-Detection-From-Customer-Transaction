# Fraud Detection Using Customer Transaction Data
Built a fraud detectiong model trained on 591k customer transactions

## Problem Statement and challenges
- This is a Fraud Detection algorithm which predicts the probability of online transactions being fraudalent and non-fraudalent.
- The class is highly unbalanced. Approximately 3% fraudalent and 97% non-fraudalent customer transactions.
- Dataset have many missing values. However, they cannot be removed because it's a sign of outliers which lead to anomily detection.
- Classical methods may not give good result, thus, ensemble method needs to be implemented to minimize the error. (Proof given below). 

## Introduction
- In this model no specific criteria has been set to classify fraudalent and non-fraudalent transactions but rather ROC curves are used to evaluate the performance.
- Model is built while keeping in mind the customer satisfaction (false positive) as well as cost of fraudalent transaction (false negative). Thus, pricision and recall are almost equally balanced.
- Model is built on data collected by IEEE Computational Intelligence Society. Consisting of approximately 600k customer transaction in training set and 500k transactions on testing set. 394 attributes are there in dataset. 
- The algorithm can be integrated in retail transaction architecture to give real-time alarms.
  
## Workflow

![Data](https://user-images.githubusercontent.com/32847030/65382701-eab09b00-dcd8-11e9-8b2a-bf08914504a6.jpg)


## Data Description


## Modeling (logistic vs RandomForest vs XGBoost)

**Down Sampling**
Down sampled the majority class to match with number of instances of minority class in order for model not to generalize towards majority class (non-fraudalent).

**Feature selection and transformation**
Converted the categorical variables into numeric using on hot encoding. In logistic regression l1 regularization is used for feature selection, random forest uses bootstraping to select subset of variables at a time and XGBoost minimizes error iteratively to build a strong classifier based on weak classifier. 

**Parameter Tuining**
Due to the size of dataset and limited computation power, the parameter tuining for XGBoost model is done manually based on calculated guesses and community sggestions. With more computation power the parameter tuning can be done automatically.   

## Results Interpretation
**Confusion Matrix and Accuracy**
Logistic Regression                      | RandomForest                                 | XGBoost
71% | 74% | 90.35%
:---------------------------------------:|:--------------------------------------------:|:---------------------------------------------:
![logistic con_mat](https://user-images.githubusercontent.com/32847030/65389708-178b9f00-dd27-11e9-99e0-c2a53e9e19d8.JPG) | ![rm con_mat](https://user-images.githubusercontent.com/32847030/65389709-178b9f00-dd27-11e9-8ec4-5fb253580e76.JPG) | ![xg con_mat](https://user-images.githubusercontent.com/32847030/65389710-178b9f00-dd27-11e9-88fd-35031da5c26e.JPG)







## Final Prediction
