import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import confusion_matrix

test = pd.read_csv("./data/test.csv")
train = pd.read_csv("./data/train.csv")

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

train.to_csv("train.csv", index=False, encoding='utf-8', index_label=True,)


# Display newly created categories
sns.countplot(x = "categories", data =train)
plt.show()


#separate features
data_labels_train = train["categories"]
data_features_train = train.drop("categories", axis=1)


# select top 8 features that are highly corrolated with sales price
data_features_train = data_features_train.filter(["SalePrice","OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd"],axis=1)

# Create train test split
features_train, features_test, labels_train, labels_test = train_test_split(data_features_train, data_labels_train, test_size=0.35, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
nb_clf = GaussianNB()

voting_clf = VotingClassifier(
    estimators=[("lr", log_clf),("rf", rnd_clf),("svc", nb_clf)],
    voting="hard"
)

voting_clf.fit(features_train,labels_train)

from sklearn.metrics import accuracy_score
for clf in (log_clf,rnd_clf,nb_clf,voting_clf):
    clf.fit(features_train,labels_train)
    y_pred = clf.predict(features_test)
    print (clf.__class__.__name__, accuracy_score(labels_test,y_pred))