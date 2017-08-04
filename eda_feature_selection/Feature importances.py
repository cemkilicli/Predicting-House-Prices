import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

test = pd.read_csv("../data/test.csv")
train = pd.read_csv("../data/train.csv")

from sklearn import preprocessing

# transform categoric labels into number
cat_labels = ['MSZoning','Street','Alley','LotShape','LandContour','Utilities',
              'LotConfig','LandSlope','Neighborhood','Condition1','Condition2',
              'BldgType','RoofStyle','HouseStyle','RoofMatl','Exterior1st',
              'Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation',
              'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
              'Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
              'Functional','FireplaceQu','GarageFinish','GarageType','GarageQual',
              'GarageCond','PavedDrive','PoolQC','Fence','MiscFeature','SaleType','SaleCondition']

le = preprocessing.LabelEncoder()
for features in cat_labels:
    le.fit(train[features])
    train[features] = le.transform(train[features])

train["SalePrice"].replace("", np.nan, inplace=True)
train = train.dropna(subset=["SalePrice"], how="all")
train = train.dropna(subset=["Id"], how="all")


print list(train)

# Create bins for categories
min_salesprice = train["SalePrice"].min() -1
max_salesprice = train["SalePrice"].max() +1
median_salesprice = train["SalePrice"].median()
low_salesprice = (min_salesprice + median_salesprice)/2
high_salesprice = (max_salesprice + median_salesprice)/2

# Create bins and categories
bins = [min_salesprice, low_salesprice, median_salesprice, high_salesprice, max_salesprice]
group_names = [1, 2, 3, 4]
categories = pd.cut(train['SalePrice'], bins, labels=group_names)
train['categories'] = pd.cut(train['SalePrice'], bins, labels=group_names)


droplabels = []
labels = list(train)

for column in labels:
    if train[column].isnull().any() == True:
        train = train.drop(column, axis =1)
        print column, "True"

print train.isnull().any()

#separate features
data_labels_train = train["categories"]

drop_labels = ["categories"]


for lables in drop_labels:
    data_features_train = train.drop(lables, axis=1)

from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)


forest.fit(data_features_train, data_labels_train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(data_features_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(data_features_train.shape[1]), importances[indices],
       color="r")
feature_names = data_features_train.columns
plt.xticks(range(data_features_train.shape[1]), feature_names, rotation=90, label='small')
plt.xlim([-1, data_features_train.shape[1]])
plt.show()