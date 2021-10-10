import pandas as pd
import numpy as np
from scipy.sparse import data
from scipy.sparse.construct import random
from helpers import eda, data_prep
from sklearn.preprocessing import RobustScaler
# Missing Values Imputation için
from sklearn.impute import KNNImputer
# titanic veri setinde bağımlı değişkenimiz (hedef) kategorik olduğundan dolayı LogisticRegression kullanacağız
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

######################################################
# * Exploratory Data Analysis
#####################################################
raw = pd.read_csv('Datasets/titanic.csv')

raw.head()

# hangi değişkende kaç adet missing value var
data_prep.missing_values_table(raw)

# gereksiz değişkenleri çıkartalım
df = raw.drop(['PassengerId','Name','Cabin','Ticket'],axis=1) # bu değişkenlerin modelimi etkilemeyeceğini düşünerek kaldırıyorum
df.head()
data_prep.missing_values_table(df)


######################################################
# * Missing Values
#####################################################
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) # Embarked değişkenini mod ile dolduruyorum
encoded_df = pd.get_dummies(df, drop_first=True) # KNN için tüm değişkenleri dönüştürdük

# Henüz aykırı değerleri analiz etmedim veya baskılamadım. Bu sebeple aykırı değerlerden etkilenmeyen RobustScaler ile değişkenleri scale ediyorum
scaler = RobustScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(encoded_df),columns=encoded_df.columns)

# 5 komşuluk birimini temel alan imputer'i oluşturdum ve missing values'i predict ile doldurulmuş new_df oluşturdum.
imputer = KNNImputer(n_neighbors=5)
new_df = pd.DataFrame(imputer.fit_transform(scaled_df), columns=scaled_df.columns)
new_df.head()

# scale edilmiş değerleri eski hallerine getirdim
new_df = pd.DataFrame(scaler.inverse_transform(new_df), columns=new_df.columns)
# Boş değerleri olan Age değişkenini yeni oluşturduğum, KNN ile boş değerleri doldurulmuş olan değişken ile değiştirdim
df['Age'] = new_df['Age']

######################################################
# * Outliers
#####################################################
df.head()
cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

# numeric değişkenlerde gez, outlier değerler varsa threshold'lar ile baskıla
for col in num_cols:
    if data_prep.check_outlier(df,col):
        data_prep.replace_with_thresholds(df,col)


######################################################
# * Logistic Regression
#####################################################
df.head()

encoded_df = pd.get_dummies(df,drop_first=True)

# * Yöntem 1: Cross-Validation
log_model = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X,y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
# Accuracy: 0.802

cv_results['test_precision'].mean()
# Precision: 0.760

cv_results['test_recall'].mean()
# Recall: 0.710

cv_results['test_f1'].mean()
# F1-score: 0.732

cv_results['test_roc_auc'].mean()
# AUC: 0.864


# * Yöntem 2: Hold-Out
X = encoded_df.drop('Survived',axis=1)
y = encoded_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=1)

log_model = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X_train,y_train)

y_pred = log_model.predict(X_test)

print(classification_report(y_test,y_pred))

# Predicting
random_person = encoded_df.sample(1, random_state=1).drop('Survived',axis=1)
random_person
log_model.predict(random_person) 









