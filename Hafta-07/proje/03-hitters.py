import pandas as pd
import numpy as np
from scipy.sparse.construct import random
from helpers import eda, data_prep
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
# * Bu projede bağımlı değişkenimiz 'Salary' olduğundan dolayı Linear Regression kullanacağız

######################################################
# * Exploratory Data Analysis
#####################################################
df = pd.read_csv('Datasets/hitters.csv')

eda.check_df(df)

data_prep.missing_values_table(df)

cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)
df.head()

data_prep.rare_analyser(df,'Salary',['League'])

######################################################
# * Missing Values
#####################################################
encoded_df = pd.get_dummies(df,drop_first=True)
scaler = RobustScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(encoded_df), columns = encoded_df.columns)
imputer = KNNImputer(n_neighbors=5)
imputed_df = pd.DataFrame(imputer.fit_transform(scaled_df), columns=scaled_df.columns)
non_missed_df = pd.DataFrame(scaler.inverse_transform(imputed_df), columns=imputed_df.columns) # standartlaştırdığımız değerleri geri kendi değerlerine (standartlaştırılmamış) döndürüyoruz
non_missed_df.head()

df['Salary'] = non_missed_df['Salary'] # orjinal veri setine impute edilmiş Salary değişkenini gönderiyoruz
data_prep.missing_values_table(df)

######################################################
# * Outliers
#####################################################
df.head()
for col in num_cols:
    print('COL NAME: ', col, "\tOUTLIERS: ", data_prep.check_outlier(df,col,0.05,0.95))

for col in num_cols:
    data_prep.replace_with_thresholds(df,col,0.05,0.95)


######################################################
# * Multiple Linear Regression
#####################################################
model_df = pd.get_dummies(df,drop_first=True)

X = model_df.drop('Salary',axis=1)
y = model_df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=6)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Eğitim Hatası
y_pred_train = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred_train))
reg_model.score(X_train, y_train)


# Test hatasi
y_pred_test = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred_test))
reg_model.score(X_test, y_test)

# Predicting
random_user = model_df.sample(1, random_state=18).drop('Salary',axis=1) # rastgele bir oyuncu seçelim

reg_model.predict(random_user) # kurduğumuz model seçtiğimiz oyuncu için ne kadar maaş tahmin etti

model_df.loc[random_user.index,'Salary'] # gerçekte seçilen oyuncunun aldığı maaş