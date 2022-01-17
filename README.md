# Credit_Risk_Analysis
Module 17: Supervised Machine Learning and Credit Risk

## Overview
This analysis will use credit card data from LendingClub to build 6 supervised machine learning models that predict credit card loan risk. As there are significantly more low-risk credit card loans than high-risk loans in this data, the models will address the imbalance in various ways. The results of those models will be compared to determine which, if any, should be used to predict credit card loan risk. 

Each model addresses the classification imbalance in a different way. 

* The ros_model uses RandomOverSampler from imblearn.over_sampling to over-sample the minority class, high-risk.

* The sm_model uses SMOTE from imblearn.over_sampling to create synthetic samples for the high-risk class. 

* The cc_model uses CLusterCentroids from imblearn.under_sampling to undersample the low-risk class. 

* The smtn_model uses SMOTEENN from imblearn.combine to combine over sampling high-risk with under sampling low-risk.

* The brfc_model uses BalancedRandomForestClassifier from imblearn.ensemble to randomly under-sample each bootstrap sample for balance.

* The eec_model uses EasyEnsembleClassifier from imblearn.ensemble to use an ensemble of AdaBoost learners that achieve balance by random under-sampling. 


## Results
The models had balanced accuracy scores between 0.54427 and 0.93166, see below. High-risk precision ranged from 0.01 to 0.09 and high-risk recall ranged from 0.61 to 0.92. Low-risk precision was 1.00 and low-risk recall ranged from 0.40 to 0.94. 

![credit_risk_model_comparison](https://github.com/JacquelineCl/Credit_Risk_Analysis/blob/aa8e0da126e24a076ce8b09310868b8cdbf3b93a/Resources/credit_risk_model_comparision.png)

## Summary
The model that performed the best at predicting true high_risk accounts was the was eec_model, with a 0.09 precision score. As this model accuratly predicts high-risk loand only 9% of the time, I would not recommend using any of these models at this time. Instead, I would recomment training of the eec_model with additional loan data. 
