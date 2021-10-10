import pandas as pd
from pandas.core.algorithms import mode
from helpers import eda, data_prep, model_evaluation
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# Bu projede de problemimiz diyabet hastası olup olmadığını tahminlemek olduğundan dolayı Logistic Regression kullanacağız. Hedef değişkenimiz 'Outcome'. 1 ise diyabet 0 ise değil

######################################################
# * Exploratory Data Analysis
#####################################################
df = pd.read_csv('Datasets/diabetes.csv')
eda.check_df(df)
data_prep.missing_values_table(df)

cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)

######################################################
# * Outliers (Local Outlier Factor)
# Bu projede diyabete özel araştırma yapamadığımdan dolayı hepsini otomatik olarak her değişkenin threshold değerlerine göre baskıladım (belki de bu hafta zamanımı çok iyi yönetemedim :/)
#####################################################
[data_prep.check_outlier(df, col) for col in num_cols] # hiç aykırı değer var mı?

for col in num_cols:
    data_prep.replace_with_thresholds(df, col)

[data_prep.check_outlier(df, col) for col in num_cols]

######################################################
# * Logistic Regression
#####################################################
scaler = RobustScaler()   
# ** doğrusal ve uzaklık ile gradient descent temelli modellerde scale etmek çok önemlidir.
df[num_cols] = scaler.fit_transform(df[num_cols]) 


X = df.drop('Outcome',axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=6)

log_model = LogisticRegression().fit(X_train,y_train)

y_pred = log_model.predict(X_test)

model_evaluation.plot_confusion_matrix(y_test,y_pred)

print(classification_report(y_test,y_pred))


# Predicting
random_person = df.sample(1, random_state=6).drop('Outcome',axis=1)
random_person
log_model.predict(random_person) 

