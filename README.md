# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset used here contains contextual information of some of the clients of a bank such as age, job, marital status, education, housing status, loan status, etc. Using this dataset, we seek to predict the reaction of the clients (positive or negative) to a marketing campaign ran by the bank. This makes it a two-class classification problem.
![Pipeline Architecture](https://github.com/Anupriya-S/Optimizing-an-ML-Pipeline-in-Azure/blob/main/Images/Screenshot%20(958).png?raw=true)

To tackle this problem we tried two different approaches:
1. Train a Scikit-learn Logistic Regression model and then optimize it using HyperDrive
2. Put Azure AutoML to job for finding the optimized model

As a result, the search power of AutoML was enough to beat the previous algorithm we used. Voting Ensemble gave the best results.

## Scikit-learn Pipeline
The classification algorithm used in this project is Logistic Regression. The steps in this Scikit-learn pipeline are as follows:
1. The dataset used is of type TabularDataset and it was created using TabularDatasetFactory class.
2. A little cleaning was performed using *clean_data()* method.
3. Then the model was trained using the default values of the hyperparameters provided in the training script.
4. After training the model, HyperDrive was used for finding the optimal values of the hyperparameters for achieving the maximum accuracy, the primary metric.
![Scikit-learn Pipeline](https://github.com/Anupriya-S/Optimizing-an-ML-Pipeline-in-Azure/blob/main/Images/Screenshot%20(943).png?raw=true)

Hyperparameters tuned:
1. Regularization Strength: optimal value --C = 0.3391
2. Max iterations: optimal value --max_iter = 90

Specifying a parameter sampler consists of two parts:
1. Defining the parameter search space: In this case it was *discrete* for "--max_iter" and *continuous* for "--C".
2. Defining the sampling method over the search space: In this case it was *Random Sampling* method.
The benefit of using *Random sampling* over any other method is that it picks up the parameters' values randomly that saves time, and the result is almost as good as any other method.

An early termination policy specifies that if you have a certain number of failures, HyperDrive will stop looking for the answer. The early stopping policy chosen here is the BanditPolicy. In this case, it basically states to check the job every two iterations. If the primary metric (Accuracy) falls outside of the top 10% range, Azure ML terminate the job. This saves us from continuing to explore hyperparameters that don't show promise helping reach our target metric.
![best_hyperdrive](https://github.com/Anupriya-S/Optimizing-an-ML-Pipeline-in-Azure/blob/main/Images/Screenshot%20(949).png?raw=true)

**After trying out various combinations of the hyperparameters, maximum Accuracy achieved by HyperDrive is 0.9089.**

## AutoML
Next we tried our hands on Automated Machine Learning or what we like to call it as *AutoML*. Basically, AutoML involves the application of DevOps principles to machine learning, in order to automate all aspects of the process (making it *MLOps*!). All the steps like feature engineering, hyperparameter selection, model training, and tuning, can be automated with the use of AutoML. With AutoML, we can get hundreds of models ready for deployment in much lesser time and even lesser efforts.
![AutoML Result](https://github.com/Anupriya-S/Optimizing-an-ML-Pipeline-in-Azure/blob/main/Images/Screenshot%20(952).png?raw=true)

**For this project, out of several models trained by AutoML (in a span of just 40 minutes) Voting Ensemble gave the highest value of accuracy, 0.9170.**

## Pipeline comparison
Its time to put these two models side by side. There is clearly a lot of difference between the two pipelines except that the end goal was to attain the maximum value of accuracy in both the cases.

Scikit-learn pipeline did this by training a Logistic Regression model and then tuning the hyperparameters associated with it using HyperDrive. In short, trained a single model and tuned the various hyperparameters and touched the highest value at **0.9089**.

On the other hand, AutoML trained and tuned 50+ models (list is attached in the output of one of the code cells in the notebook) in a short span of just 40 minutes and was able to push the accuracy up to **0.9170**.

## Future work
As they say, there is isn't anything like a last commit to a repository, we can always try something new. So here are some of the things that we can try to achieve better results.

For using HyperDrive:
1. Dataset should be balanced before feeding it to the model for getting better results.
2. We can alter the parameter sampler for trying out new combinations of hyperparameter values.
3. In HyperDriveConfig, increase the value of *max_total_runs*.
4. Use a different train test split ratio.
5. And of course, a new classification algorithm with different hyperparameters can be used for the job at hand.

For using AutoML:
1. In AutoMLConfig, increase *experiment_timeout_minutes* so that more possibilities can be explored.
2. Set the value of *iterations* as per the requirement.
3. Modify the value of *n_cross_validations* and much more.

Even this is not the limit. There must be many other ideas to beat the best of our results, **0.9170**.
