# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
+ The dataset on this project contains data about direct marketing campaigns through a phone call of a banking institution. This project aims to predict if the client will subscribe `(yes/no)` to a term deposit `(y variable)`.

+ The best performing model is the Voting Ensemble Classification model executed during AutoML run. It's accuracy score is `91.55 %` while the Sci-kit learn Logistic Regression model scored a `91.09 %` accuracy. 

## Scikit-learn Pipeline
+ After initializing the workspace and creating a compute target for model training, the first step in the project is to create a `TabularDataset` from a CSV file downloaded from a public web URL using the `TabularDatasetFactory` class.

+ The data was then *prepared*, *cleaned*, and *split* using the pre-configured training script **'(train.py)'**. The data is categorically sorted and thus *one-hot encoding* is used to clean up the data executed within the 'clean data' function in the 'train.py' script. One-hot encoding creates a new binary column that shows the presence of each potential value from the original data. The dataset was then split into a *train (2/3)* and a *test (1/3)* dataset.

+ The classifying algorithm used is **logistic regression**. It is used to estimate discrete values (yes / no, 0 or 1, true / false) based on a set of independent variables. This is a special case of linear regression where the outcome variable is categorical. 

+ Next, **SKLearn estimator** was constructed. This estimator will provide a simple way of deploying the training job on the compute target. 

+ Using the Python SDK we start the hyperparameter tuning run and chose **Random Sampling Parameter** strategy.  The ranges for the *inverse of the regularization strength* and choices for *maximum number of iterations to converge* are provided. Then **Bandit Policy** was used to describe an early termination policy for the termination of jobs that are not performing well.

+ **HyperDriveConfig** was then configured and `Accuracy` was specified as the primary metric. The maximum number of concurrent jobs is set to 4 which is equal to the number of nodes in the compute cluster. 

+ Lastly, the hyperdrive run was submitted to the experiment and the model from the *best run* was saved.

**What are the benefits of the parameter sampler you chose?**
+ **Hyperparameters** are customizable parameters that you select for model training that guide the training process. **Random Parameter Sampling** is used to describe random sampling over the search space of a hyperparameter.

+ In Scikit-learn LogisticRegression model, **Random Parameter Sampling** is used to try different configuration sets in the distribution of continuous `(--C)` and choice among discrete `(max_iter)` hyperparameters that will maximize the *primary metric `(Accuracy)`*.

**What are the benefits of the early stopping policy you chose?**
+ The **Bandit Policy** basically states to verify the job *for every 2 iterations*. If the primary metric (accuracy) falls below the *top 10 % range*, Azure ML will **kill** the job. This saves the consumption of a lot of computational resources and from continuing to explore hyperparameters that don't show promise of helping reach our target metric.
## AutoML
+ Automated machine learning, also referred to as automated ML or AutoML, is the method of automating the time-consuming, iterative tasks of machine learning model creation. It helps data scientists, analysts and developers to create high-scale, effective and profitable ML models while preserving the consistency of the model. 

+ The AutoML run provided a lof of algorithm models executed before the specified timeout policy ends. The most notable models it implemented are `Stack ensemble`, `MaxAbsScaler XGBoostClassifier`, and `VotingEnsemble`. Of all the models deployed `Voting Ensemble` outputs the highest scored accuracy. The hyperparamaters used by AutoML is configured automatically by the machine depending on the model algorithm that is being test to maximize the primary metric defined `(accuracy)`. 

+ Also, `AutoML_Config` class should be configured for the submission of an automated ML experiment in Azure Machine Learning.This configuration object contains and manages the parameters for the configuration of the experiment run, as well as the training data to be used at run time. Some of these parameters are `primary_metric`(the automated machine learning metric will be optimized for model selection.) , `n_cross_validations`(how many cross validations can be performed when the user's validation data is not known), and `experiment_timeout_minutes`(total time in minutes that all iterations combined will take until the completion of the experiment)

## Pipeline comparison
+ There are no significant difference in the accuracy results between the hyperdrive and autoML run. Of the two, the best performing is the AutoML, courtesy of its `91.55 %` accuracy obtained by the Voting Ensemble classification model which is just `0.46 %` higher than the `91.09 %` of the best run of Logistics Regression model in the hyperdrive run. This result is imperative as AutoML runs all the available machine learning algorithms for the available dataset while hyperdrive optimizes only the hyperparameters of a single Scikit learning algorithm model to get the best result. 

+ AutoML and Scikit-learn pipeline architecture differs in the ingestion and preparing of the dataset before it is passed on training job. In Scikit-learn, training script is used to fetch the data and the hyperparameters. In AutoML, all the pipeline steps happened inside the Jupyter Notebook.

## Future work
+ During the autoML run, the dataset didn't pass the *class balancing detection data guardrail*. It tells that the input data has a bias against one class and such this unbalanced data may contribute to wrongly positive impact on the accuracy of the model. Using a performance metric that better manages the imbalanced data can possibly fix this. For example, `AUC weighted` is a primary metric that measures the contribution of each class on the basis of the relative number of samples representing that class, making it more robust against imbalance.

+ The accuracy of the primary metric in the Linear Progression model used in Hyperdrive optimization is very good, but there are still a lot of things that can be used to boost accuracy. *Normalization* and *regularisation* of data are two of the multiple ways to gain improvements. Normalizing data (i.e., shift it to have a mean of zero, and a spread of 1 standard deviation) will result to fewer null values and less redundant data, making the dataset more compact. Regularization of features, on the other hand, tends to minimize potential over-fitting.

## Proof of cluster clean up
**Image of cluster marked for deletion**
![alt text](https://github.com/UnhelpfulRascal/AzureML/blob/main/compute_delete.png)
