import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
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

"""
# Display newly created categories
sns.countplot(x="categories", data=train)
plt.show()
"""

# separate features
data_labels_train = train["categories"]
data_features_train = train.drop("categories", axis=1)
data_features_train = data_features_train.drop("Id", axis=1)
data_features_train = data_features_train.drop("SalePrice", axis=1)

columns = data_features_train.columns


# Handle missing values in Training Data Set
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data_features_train)
data_features_train = imp.transform(data_features_train)

# Create train test split
features_train, features_test, labels_train, labels_test = train_test_split(data_features_train, data_labels_train, test_size=0.25, random_state=42)

from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=350,
                              criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=2,
                              min_weight_fraction_leaf=0.0, max_features="sqrt", max_leaf_nodes=None,
                              min_impurity_split=1e-07, bootstrap=False, oob_score=False, n_jobs=1,
                              verbose=0, warm_start=True, class_weight=None)


forest.fit(features_train, labels_train)
pred = forest.predict(features_test)

importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


from sklearn.metrics import accuracy_score
print "accuracy is", accuracy_score(labels_test, pred)
print "root mean squared error is", sqrt(mean_squared_error(labels_test, pred))
from sklearn.metrics import r2_score
print "R-squared Error",r2_score(pred,labels_test)

from sklearn.metrics import confusion_matrix
print confusion_matrix(labels_test, pred)


# Print the feature ranking
print("Feature ranking:")
for f in range(data_features_train.shape[1]):
    print columns[indices[f]], importances[indices[f]]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(data_features_train.shape[1]), importances[indices],
       color="b")
plt.xticks(range(data_features_train.shape[1]), columns[indices], rotation=90, label='small')
plt.xlim([-1, data_features_train.shape[1]])
plt.show()
