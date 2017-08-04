import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
from math import sqrt


train = pd.read_csv("../data/train.csv")

from sklearn import preprocessing

# transform categoric labels into number
cat_labels = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
              'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
              'BldgType', 'RoofStyle', 'HouseStyle', 'RoofMatl', 'Exterior1st',
              'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
              'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
              'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
              'Functional', 'FireplaceQu', 'GarageFinish', 'GarageType', 'GarageQual',
              'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

le = preprocessing.LabelEncoder()
for features in cat_labels:
    le.fit(train[features])
    train[features] = le.transform(train[features])

train["SalePrice"].replace("", np.nan, inplace=True)
train = train.dropna(subset=["SalePrice"], how="all")

# Create bins for categories
min_salesprice = train["SalePrice"].min() - 1
max_salesprice = train["SalePrice"].max() + 1
mean_salesprice = train["SalePrice"].median()
low_salesprice = train["SalePrice"].quantile(q=0.25)
high_salesprice = train["SalePrice"].quantile(q=0.75)

# Create bins and categories
bins = [min_salesprice, low_salesprice, mean_salesprice, high_salesprice, max_salesprice]
group_names = [1, 2, 3, 4]
categories = pd.cut(train['SalePrice'], bins, labels=group_names)
train['categories'] = pd.cut(train['SalePrice'], bins, labels=group_names)

# Display newly created categories
sns.countplot(x="categories", data=train)
plt.show()

#find null values
for column in train.columns:
    if train[column].isnull().any() == True:
        column_mean = train[column].mean()
        print column_mean
        train[column].replace("", np.nan, inplace=True)
        train[column] = train[column].fillna(column_mean)

# Train, Validate, Test Split
data_train, data_validate, data_test = np.split(train.sample(frac=1), [int(.6*len(train)), int(.8*len(train))])

# separate features
labels_train = data_train["categories"]
features_train = data_train.drop("categories", axis=1)

labels_validate = data_train["categories"]
features_validate = data_train.drop("categories", axis=1)

labels_test = data_train["categories"]
features_test = data_train.drop("categories", axis=1)

# Train model
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier( n_estimators=300, max_features="auto")
rnd_clf.fit(features_train, labels_train)



# validate prediction
pred_validate = rnd_clf.predict(features_validate)
print " "
print "Validate Scores"
print "Accuracy is:", accuracy_score(labels_validate, pred_validate)
print "RMSE is:", sqrt(mean_squared_error(labels_validate, pred_validate))

# test prediction
pred_test = rnd_clf.predict(features_test)
print " "
print "Test Scores"
print "Accuracy is:", accuracy_score(labels_test, pred_test)
print "RMSE is:", sqrt(mean_squared_error(labels_test, pred_test))
