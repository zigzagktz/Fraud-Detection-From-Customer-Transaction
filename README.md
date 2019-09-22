# Fraud Detection Using Customer Transaction Data
Python, Scikit-learn, Machine Learning

## Problem Statement and challenges
- This is a Fraud Detection algorithm which predicts the probability of online transactions being fraudalent and non-fraudalent.
- In training data class is highly unbalanced. Approximately 3% fraudalent and 97% non-fraudalent customer transactions.
- Dataset have many missing values. However, they cannot be removed because it's a sign of outliers which lead to anomily detection.
- Classical methods may not give good result, thus, ensemble method needs to be implemented to minimize the error. (Proof given below). 

## Introduction
- In this model no specific criteria has been set to classify fraudalent and non-fraudalent transactions but rather ROC curves are used to evaluate the performance.
- Model is built while keeping in mind the customer satisfaction (false positive) as well as cost of fraudalent transaction (false negative). Thus, we care about better values of precision and recall.
- Model is built on data collected by IEEE Computational Intelligence Society. Consisting of approximately 600k customer transaction in training set and 500k transactions on testing set. 394 attributes are there in dataset. 
- The algorithm can be integrated in retail transaction architecture to give real-time alarms.
  
## Workflow

![Data](https://user-images.githubusercontent.com/32847030/65382701-eab09b00-dcd8-11e9-8b2a-bf08914504a6.jpg)


## Data Description
![data dim](https://user-images.githubusercontent.com/32847030/65395115-7de2e280-dd64-11e9-853d-1b179ec7ff44.JPG)

TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
TransactionAMT: transaction payment amount in USD
ProductCD: product code, the product for each transaction
card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
addr: address
dist: distance
P_ and (R__) emaildomain: purchaser and recipient email domain
C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
D1-D15: timedelta, such as days between previous transaction, etc.
M1-M9: match, such as names on card and address, etc.
Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

## Modeling (logistic vs RandomForest vs XGBoost)

**Down Sampling**
Down sampled the majority class to match with number of instances of minority class in order for model not to generalize towards majority class (non-fraudalent).

**Feature selection and transformation**
Converted the categorical variables into numeric using on hot encoding. In logistic regression l1 regularization is used for feature selection, random forest uses bootstraping to select subset of variables at a time and XGBoost minimizes error iteratively to build a strong classifier based on weak classifier. 

**Parameter Tuining**
Due to the size of dataset and limited computation power, the parameter tuining for XGBoost model is done manually based on calculated guesses and community sggestions. With more computation power the parameter tuning can be done automatically.   

## Results Interpretation
**1 : Confusion Matrix and Accuracy**

Logistic Regression                      | RandomForest                                 | XGBoost
:---------------------------------------:|:--------------------------------------------:|:---------------------------------------------:
![logistic con_mat](https://user-images.githubusercontent.com/32847030/65394133-b92ae480-dd57-11e9-9faf-459f19a4b439.JPG) | ![rm con_mat](https://user-images.githubusercontent.com/32847030/65394134-b92ae480-dd57-11e9-9127-724b1eda8159.JPG) | ![xg con_mat](https://user-images.githubusercontent.com/32847030/65394255-402c8c80-dd59-11e9-9cbc-7a288cfd04fc.JPG)
70.42% | 74.7% | 90.19%


- Logistic regression has given more ***Flase positive*** than randomforest. Allowing more frauds to go undetected, hence less vigilent.
- Randomforest has given more ***Flase negative*** than Logistic. Picking up many false alarms, hence reduced customer satisfaction.
- XGB giving best accuracy.
                         
              
              
              
**2 : Area Under the Curve with ROC curve**       

Logistic Regression                      | RandomForest                                 | XGBoost
:---------------------------------------:|:--------------------------------------------:|:---------------------------------------------:
![logistic ROC](https://user-images.githubusercontent.com/32847030/65394172-32c2d280-dd58-11e9-91b6-efea6251ff5b.JPG) | ![rm roc](https://user-images.githubusercontent.com/32847030/65394173-32c2d280-dd58-11e9-8cb0-a374add9b837.JPG) | ![xg roc](https://user-images.githubusercontent.com/32847030/65394260-4ae72180-dd59-11e9-9fd9-8b352924e132.JPG)

Why do we need this?
> Because simple classification can only consider one therashold for classifying two classes. For example less than 50% class 0 and more than 50% class 1. However, ROC gives proper understanding of overall distribution of class probability. 

- False Positive Rate: False Positive / (False Positive + True Negative) 
- True Positive Rate: True Positive / (True Positive + False Negative)

The desired value of ROC curve is when it reach top left corner of the graph. Meaning XGB outperforms both logistic and randomforest




**3 : Precision, Recall and F1 score**       

Logistic Regression                      | RandomForest                                 | XGBoost
:---------------------------------------:|:--------------------------------------------:|:---------------------------------------------:
![logistic report](https://user-images.githubusercontent.com/32847030/65394296-ed070980-dd59-11e9-9656-8533a1565c04.JPG) | ![rm report](https://user-images.githubusercontent.com/32847030/65394297-ed9fa000-dd59-11e9-8dc7-d149ab64d3fd.JPG) | ![xg report](https://user-images.githubusercontent.com/32847030/65394298-ed9fa000-dd59-11e9-8437-09d02ed47b0c.JPG)

Very important distinction here is that even though randomforest have better accuracy than logistic regression, but it is performing poorly on detecting frauds which is evident by low precision value of 1.

- Randomforest is better at defining what transactions are not fraudalent (poor precision for class 1)
- Logistic regression is better at defining what transaction are fraudalent (better precision for class 1)


## Final Prediction

Final prediction is saved into a cvs file with probability associated with each transaction of being fraudalent. Only two fields are shown below, the transaction ID and probability to give clearity in result

![final](https://user-images.githubusercontent.com/32847030/65394857-9e10a280-dd60-11e9-9113-67e7ce1ae40b.JPG)

