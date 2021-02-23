# **Prediction of Cross Sales of Health Insurance 🏠 🏥** #
## ***"Made as a final project task at Rakamin Data Science Bootcamp"*** ##
**by: Robertsen Putra Sugianto**

### **- Background**

India roughly accounts for only about 1% of the global vehicle population. However, it accounts for about 6% of total global road accidents 
1. Based on reports from insurance regulators in India in 2018 the vehicle (motorbike) insurance business increased by 8.91% 
2. As an insurance company that has developed data on health insurance owners, it is hoped that this data can be used for the vehicle insurance business. In the process of developing a vehicle insurance business, building a model to predict whether health insurance users will be interested or not.

**Business Domain**: Vehicle Insurance

**Problem**: How to predict whether a health insurance owner will be interested in vehicle insurance?

**Business Metrics**: Number of health insurance customers who use Vehicle Insurance, Revenue, and Win Rate

**Goals**: Predict the interest of health insurance owners in Vehicle Insurance to increase Revenue, Win Rate, and number of customers

**Objective**: Build a model to predict whether health insurance owners will be interested in vehicle insurance from data from health insurance owners and their responses.

**Predictive Analytics**: Creating a model that can help predict whether customers, including those who will be interested in vehicle insurance or not


### **- Dataset**

- The datasets needed in making this model can be obtained from **the internal record of Health Insurance owner data** which contains customer data including Demographics (Gender, Age, Region Code), Vehicles (Vehicle Age, Vehicle Damage), Policy (Annual Premium, Policy Sales Channel). ), etc. ***(https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction)***

- This dataset has a total of **12 columns**, 381,109 rows of data, and no data is lost. With **3 columns of which are categorical** (Gender, Vehicle_Age, Vehicle_Damage), and the remaining **8 columns are numeric**.
- In general, there is no statistically strange data, with the exception of the Annual_Premium column which has a very large maximum value and is very different from the minimum value.
- There are quite a lot of **outliers** for the Annual_Premium column, with quite a large number and a long way off
- The Age distribution is **Positively Skewed**. The response is also quite **imbalanced**
- Vehicle_Damage has a fairly balanced data distribution. Meanwhile, Vehicle_Age is quite lame, with the category value> 2 Years far behind the others.


### **- Insight**
<p align="center">
<img src="https://github.com/robertsenputras/DS-HealthInsuranceCrossSelling/blob/master/fig/Insight%20-%20Win%20Rate%20before.png"  width="400" height="300">
</p>
- People whose vehicles have been damaged have the potential to take out insurance. The bad experience someone has about their vehicle breaking makes them think about taking out insurance
- Customers with older vehicles are also more interested in vehicle insurance than customers with younger vehicle ages
- People in Region Code 28 are a suitable target to increase the number of customers, as many are attracted to vehicle insurance offers, far more than other regions.
- The level of insurance acceptance in Region Code 28 is the highest, reaching 23.03% followed by Region Code 29, then 41, and 11.
- The number of people who insure their vehicles in Region 28 (Tamil Nadu), is because people in Region 28 have their vehicles damaged.
- Potential buyers are in the age range of 30 to 62 years

Preprocessing Data
In this dataset, there are no missing values
In this dataset, there are no duplicate data.
The data of Annual Premium was normalized with the log function because the data was very large and so that it was normalized.
Detect and remove outliers with a range in the form of IQR. Annual Premium data is data whose outliers are discarded because other data does not have outliers

### **- Feature Engineering**
Merge the value of '> 2 Years' with the value '1-2 Years' to '> 1 Year' because the value of '> 2 Years' is far behind (lame).
Age, Annual_Premium, Policy_Sales_Channel, and Vintage features are standardized with the StandardScaler () function. Normalization is not used because it is more universal. This process needs to be done to overcome the column with a large value that will have a large effect on the results.
Perform encoding features for Gender, Vehicle_Age, and Vehicle Damage features using pandas' get_dummies function. The encoding process uses the drop_first = True parameter because there are only 2 columns that are correlated with one another.
Change the data type of Region_Code feature which was previously float data type to string for binary encoding purposes.
Doing BinaryEncoding of the Region_Code feature which has a unique value of 53 to 7 classes then reduces it to 6 classes because one feature is only worth 0. One hot is not used because this method will generate 53 new columns.
Perform Class Imbalance handling with under_sampling so that the total data is reduced from 300 thousand data to below 100 thousand data.


### **- Modeling**
#### - Evaluation Metrics

The AUC (Area Under Curve) metrics are used. This is because this is a case of ranking problems which is very suitable to be implemented. The data is ranked based on the probability that people will take vehicle insurance.

![Comparison between models](https://github.com/robertsenputras/DS-HealthInsuranceCrossSelling/blob/master/fig/persen.png)
Besides AUC, metric precision is also used.

We use precision because precision is suitable when we are concerned about the win rate / customer conversion ratio. The size of the winrate will reduce offering cost and will increase revenue due to the effectiveness of the sales team's offer.
The greater the value of the precision produced by the model means the more likely it is to find people who will actually take out vehicle insurance.


### **- Modeling stage**

Determine the baseline model
 which will be used
The determination process is done by using RandomizedSearch for Hyperparameter tuning in the algorithm: Logistic Regression, KNN, NB, ANN, Decision Tree, Random Forest, XGBoost, AdaBoost.

I think XGBoost is the best model

Full feature engineering is used at this stage.


Iterates feature engineering on the XGBoost model
Feature iterations that are done:
One Hot Encoding only (imbalance)
One Hot Encoding + Outliers Handling (imbalance)
One Hot Encoding (oversampling)
One Hot Encoding (undersampling)
One Hot Encoding + standardization + outliers handling (undersampling)

Obtained the best engineering features occurred during one hot encoding + standardization + outliers handling and download sampling, as evidenced by the results of submitting test data on Vidhya's website analytics, namely 74.8%

The ROC results on the train data in the imbalance class are around 84%, and the ROC results on the undersampling data are around 85-86%. And the results of the train on data that have been sampled are around 90%.
To prevent overfitting, evaluate the test data by submitting it to https://datahack.analyticsvidhya.com/contest/janatahack-cross-sell-prediction


Based on precision and AUC values. Then the chosen model is XGBoost

### **- Hyperparameter tuning**
After obtaining the best engineering features, hyperparameter tuning was performed again


### **- Validate with cross-validation**
Split the data into 3 parts, namely train data, validation data, and test data.

### **- Cross-validation**
StratifiedKfolds (with k = 10) are used to ensure the distribution of each class in the train and validation is the same


### **- Importance Feature**
From the results of a more in-depth analysis of feature importance, we can find out that people who tend to take out vehicle insurance are people who:
- the vehicle was damaged before
- Never took out vehicle insurance before
- the age of the vehicle more than one year.

### **- Recommendation**
Use the proposed model to predict potential customer
Focusing marketing in Region Code 28 (Tamil Nadu) by adding more store or join an exhibition
Reach people who are 30-62 years of age by giving any promotion / dicsount for millennials, and easy apply to x generation
Reach out to people who have damaged their car and old car age by collaborating with second hand dealer and garage
