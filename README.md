# The Exploration of Modern Slavery
---

The exploration of modern slavery poses a significant and urgent challenge that necessitates the use of advanced analytical methods and state-of-the-art technologies. As such, the examination of modern slavery has been designated as the second assignment in the Analytics Practicum I (Python) course, which forms an integral part of the MSc in Business Analytics program offered by the Department of Management Science and Technology at the Athens University of Economics and Business.

The current assignment has been inspired by one study, that has appeared in the journal [Nature](https://www.nature.com/). The authors - researchers of the study, have been trying to estimate the prevalence of modern slavery, as well as factors that can help with this prediction. Machine Learning techniques can help in this endeavour, as has been shown through the following study:

* [Lavelle-Hill, R., Smith, G., Mazumder, A. et al. Machine learning methods for "wicked" problems: exploring the complex drivers of modern slavery. Humanities and Social Sciences Communications 8, 274 (2021)](https://doi.org/10.1057/s41599-021-00938-z)

More specifically, in the above-mentioned paper titled "Machine learning methods for 'wicked' problems: exploring the complex drivers of modern slavery" Lavelle-Hill et al. aim to explore the complex drivers of modern slavery by using machine learning methods to identify patterns and predictors of modern slavery.

Through the abstract of the paper we can identify that **modern slavery** is a complex and challenging issue to study, with data scarcity and high dimensionality impeding efforts to isolate and assess the importance of individual drivers statistically. However, recent advances in non-parametric computational methods offer scope to better capture the complex nature of modern slavery. In a study that models the prevalence of slavery in 48 countries, non-linear machine-learning models and strict cross-validation methods were combined with novel variable importance techniques, emphasising the importance of stability of model explanations via a Rashomon-set analysis.

The results of the study highlighted the importance of new predictive factors, such as a country's capacity to protect the physical security of women, which had been previously under-emphasised in quantitative models. Further analyses uncovered that women are particularly vulnerable to exploitation in areas where there is poor access to resources. The model was then leveraged to produce new out-of-sample estimates of slavery prevalence for countries where no survey data currently exists, providing valuable insights for policymakers and researchers seeking to combat modern slavery. Overall, this study demonstrates the potential of machine learning methods to identify patterns and predictors of complex social problems such as modern slavery.

While the current analysis will be done on [Jupyter Notebook](http://jupyter.org/) and in [Python 3.10.0](https://www.python.org/downloads/release/python-3100/).
 
---

> Dimitrios Matsanganis <br />
> dmatsanganis@gmail.com, dim.matsanganis@aueb.gr

---

---
## Question 1: Data Preprocessing

---

To begin with, the original dataset used in the analysis of modern slavery contained a single dependent variable, namely, country-level slavery prevalence estimates expressed as a percentage of the population. The dataset covered a total of 48 countries and included prevalence estimates for the years 2016 and 2018.

However, the dataset also contained missing values, which the researchers addressed through the use of assumptions and imputation methods. For the purposes of this assignment, we will start with the unimputed data and apply the same assumptions taken by the researchers to fill in the missing values.

In addition to this, we will also leverage the imputation methods provided by the [scikit-learn](https://scikit-learn.org/stable/) library to ensure that our dataset is complete and ready for analysis. By taking these steps, we can ensure that our analysis is based on accurate and complete data, which will enable us to derive meaningful insights and actionable recommendations related to the drivers of modern slavery, guided from the researcher's study.

The initial preparatory step of this assignment is the importation of the Python libraries that will be used in the following questions of the assignments.
These libraries are presented below:

* [pandas](https://pandas.pydata.org/docs/)
* [numpy](https://numpy.org/)
* [os](https://docs.python.org/3/library/os.html)
* [matplotlib](https://matplotlib.org/)
* [sklearn](https://scikit-learn.org/stable/)
* [statsmodels](https://www.statsmodels.org/stable/index.html)
* [scipy](https://scipy.org/)

**Note!**
The aforementioned libraries, in order to function correctly, must be installed in Python **locally before being imported**.

Some notes for the dataset:

* The dataset comprises 70 rows and 120 columns, with the first two columns, namely `Data_year` and `Country`, serving as index columns for each observation. These index columns determine the country-level and year-group (2016 or 2018) to which each observation corresponds. The dataset also includes a `Region` column, represented by other columns used as dummy variables, that categorizes each country according to its broad geographical area.


* It should be noted that not all countries have observations for both year groups. For instance, Afghanistan and Argentina have observations only for the year 2018, Brazil and Bolivia only for 2016, while Botswana and Cambodia has observations for both the years 2016 and 2018.


* Furthermore, the dataset includes exogenous variables, some of which indicate the year in which their data were collected, while others do not.


* From the researcher papers and through the dataframe we observe that some columns tend to have a year at the name (e.g. AIDS_Orph_2016) but have variables for 2018 group of years (>2016). 


* More specifically, the following code sets the values of certain cells to `0` for variables that exist for both 2016 and 2018, depending on the year group of the corresponding country. Specifically, the first loop sets the values of `AIDS_death_2016`, `AIDS_Orph_2016`, and `Phys_secF_2014` to `0` for rows where `Data_year` is equal to 2018. The second loop sets the values of `AIDS_death_2018`, `AIDS_Orph_2018`, and `Phys_secF_2019` to `0` for rows where `Data_year` is equal to 2016.


* To further clarify, these changes are specific to rows where `Data_year` is either 2016 or 2018 and only apply to the columns that represent the same variable for both years, but actually belong only to one year group.


---
## Question 2: Slavery Estimation Using All Features

--- 

This task involves the training of multiple models, including linear regression models, decision trees, and random forests, to predict slavery prevalence using all the available features in the dataset. The imputed data will be utilized to train these models, separately for predicting slavery prevalence in 2016 and 2018. 


The evaluation of model performance will be conducted using the Mean Average Error (MAE) metric on the [Out of Sample data](oos_data.csv). Moreover, the importance of various features in predicting slavery prevalence can be assessed using ridge regression and lasso regression for linear regression. Such analysis can help identify critical factors that contribute significantly to the prevalence of slavery in different regions, which can facilitate the development of effective policies to combat this social issue.

In this point we need to highlight that we need to create **six** models, two per category (two linear regression models, two decision trees, and two random forests) for each of the two year groups (2016 and 2018).

### Linear Regression - Ridge/Lasso Regression
---

Linear regression is a commonly used statistical method for predicting a continuous outcome based on a set of predictor variables. However, when the number of predictor variables is high, or when the predictors are highly correlated, linear regression models can suffer from overfitting or instability.


To address these issues, alternative regression techniques such as ridge regression and lasso regression have been developed. These methods are particularly useful when dealing with high-dimensional data and can help to identify the most important predictors for the outcome of interest.


In this context, one application of linear regression models could be to study the factors that contribute to the prevalence of slavery in different countries. By using ridge and lasso regression, it is possible to identify the most important features that affect slavery prevalence, which can then inform policy decisions aimed at preventing and reducing slavery.


The following code sets up a k-fold cross-validation process, where the data is split into k equal subsets and each subset is used once as a validation set while the other k-1 subsets are used for training. The Lasso regression algorithm is used to fit a model on the training set, with different values of alpha regularization parameter specified in the alpha_values list. The negative mean squared error is used as the evaluation metric to assess the performance of the model. The loop iterates through each value of alpha in the alpha_values list, and for each alpha value, the Lasso model is fitted and evaluated using the cross_val_score function. The negative mean squared error score for each fold is averaged across all folds to obtain the validation score for that alpha value. The validation_scores list is then populated with the average validation scores for each alpha value. Finally, best_alpha16 is assigned the alpha value with the highest average validation score, which is the optimal value of alpha that results in the best model performance for the 2016 dataframe.

### Decision Tree Models
---

Decision tree models are a popular class of machine learning models used for both classification and regression tasks. Decision trees are constructed by recursively partitioning the feature space into regions that correspond to different predicted outcomes. Each partitioning decision is based on a threshold value for a particular feature, and the final output of the model is determined by the predicted outcome associated with the region of the feature space that the test point falls into. Decision trees can be used for both simple and complex datasets and are relatively interpretable compared to other machine learning models.

Through the papers we found out that the tree's depth shold be from 3 to 5 ([3, 5]). The authors maybe avoid to use largest trees to avoid overfitting to the existing data and the time-consuming issues that present with the largest trees.

Thus, we will find the best estimator and parameters for each Decision Tree model for each of the two group years dataframes.


### Random Forest models
---

Random Forest is a popular ensemble learning method for regression tasks that combines multiple decision trees to produce more accurate predictions. In Random Forest, each decision tree in the ensemble is built on a random subset of the features and the rows of the training data, which helps to reduce overfitting and increase the generalizability of the model.

To evaluate the performance of Random Forest models, we will use Mean Absolute Error (MAE), which measures the average absolute difference between the actual and predicted values of the target variable. The lower the MAE, the better the model's performance.

We will start with the 2016 data and we will construct the Random Forest models to try to find the best combination of parameters and then we will move forward to 2018 Random Forest models.

We initially create the random forest model with 40 estimators (trees) and default internal split nodes for the group year 2016.


---
## Question 3: Slavery Estimation with Theory-based Features

---

The researchers in Question 2 utilized the full model with the 106 features. In this question the initial 106 features were reduced to a subset of 35 features to train the models. The models will be trained using the reduced feature set, and the evaluation was performed using the Out of Sample data and through the MAE (Mean Absolute Error), which is a metric used to evaluate the accuracy of a regression model.

Upon analyzing the results, it was observed that some features were more important than others in predicting slavery prevalence. The feature importance can be used to determine which features have the most impact on the model's performance.

Through our investigation we found out that these 35 variables are the followings:

'KOF_Globalis', 'Work_rightCIRI', 'Trade_open', 'GDPpc',
'Armedcon', 'Poverty', 'Stunting_u5s', 'Undernourish', 'Wasting_u5s', 'Maternal_mort',
'Neonatal_mort', 'Literacy_15_24yrs', 'F_school', 'ATMs', 'Child_lab', 'Unemploy', 'Infrastruct',
'Internet_use', 'Broadband', 'Climate_chg_vuln', 'CPI', 'Minority_rule', 'Freemv_M',
'Freemv_F', 'Free_discuss', 'Soc_powerdist', 'Democ',
'Sexwrk_condom', 'Sexwrk_Syphilis', 'AIDS_Orph', 'Rape_report',
'Rape_enclave', 'Phys_secF', 'Gender_equal'.

These variables found at the paper's Github and more specifically at the [Rashomon_Variable_Importance.py](https://github.com/ml-slavery/ml-slavery/blob/main/Rashomon_Variable_Importance.py).

---
## Question 4: Slavery Estimation with PCA-derived Features

---

To reduce the dimensionality of the data and simplify the modeling process, we will use Principal Component Analysis (PCA) to reduce the full set of features to six. PCA is a statistical technique that transforms a high-dimensional dataset into a lower-dimensional space while retaining as much of the original variance as possible.

After applying PCA, we will obtain six derived features. These features can be interpreted as linear combinations of the original features that capture the most significant information. We will then train and evaluate our models on the PCA-derived features. This will allow us to determine the effectiveness of using the reduced feature set for predicting slavery prevalence.

To be more precise, in this Question we are applying PCA to the initial dataset of 103 variables in order to identify the most important variables that contribute to the prediction of slavery prevalence. We are selecting six derived features to use as inputs for our models, which capture the most significant information in the original dataset.

The first step of this procedure is to efficiently apply the PCA method multiple times, we can construct a pipeline of steps that will allow us to automate the process. This pipeline can be used to reduce the dimensions of both the group years datasets (2016 and 2018) and their corresponding test datasets. Since we will need to perform PCA four times, this approach will save us time and help to ensure consistency in the application of the method.

By constructing a pipeline, we can streamline the process of applying PCA and ensure that each step is performed in a consistent and standardized way. This will make it easier to replicate the analysis and compare the results across different datasets and models.

---
## Summary of the Assignment
---

The results indicate that using a reduced dataset obtained through PCA, with only six principal components, may not be sufficient to capture the variability present in the original datasets. This is suggested by the lower predictive performance (MAE) of the previous machine learning models trained on the reduced datasets or the full set compared to those through PCA (best MAE was 0.31 with PCA is 0.356).

To be more precise, the Random Forest model for the full model of 2016 and the reduced one (the one with the theory based features) of 2016, output models with MAE of 0.31, while the best MAE for the 2016 dataset after the PCA implementation was again from the Random Forest model but with 0.45 value of MAE. 

Regarding the 2018 dataset the best MAE origins from the reduced 35-variable dataset with a MAE of 0.32. However, all Random Forest implementations had similar ranged MAEs with 0.34 on the full model and 0.36 after the PCA implementation.

From the aforementioned we can summarize that the differences of MAE scores can be considered big only at the 2016 data model and not on the 2018 data model. One assumption can be made that since the 2018 dataset contains 45 observations/countries and 2016 dataset only 25, the size of the datasets can be the factor for these different results regarding the models' MAE.

Finally, the best model of our research is the Random Forest one for the 2016 data with 0.31, followed by the Random Forest of the full model with 0.315.
