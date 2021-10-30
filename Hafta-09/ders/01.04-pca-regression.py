
################################
# * BONUS
################################
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


################################
# * BONUS: Principal Component Regression
################################
# * Önce PCA modelini kuruyorum.
def get_scaled_df():
    df = pd.read_csv("datasets/Hitters.csv")
    num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col] # Salary haricindeki numeric değişkenleri seçiyorum.
    df = df[num_cols]
    df.dropna(inplace=True)
    df = StandardScaler().fit_transform(df) # numerik değişkenleri scale edip, değişkene atıyoruz (numpy arraylerden oluşan)
    return df

df = get_scaled_df()
pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df) # sayısal değişkenleri 3 boyuta indirgedik


# * Kategorik değişkenler ile bileşenleri birleştireceğim.
df = pd.read_csv("datasets/Hitters.csv")
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"]).head()

df[others].head()


final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                      df[others]], axis=1)

final_df.head()

# * Encoder ile kategorik değişkenleri encode edeceğim.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)


final_df.dropna(inplace=True)


y = final_df["Salary"] # bağımlı değişken
X = final_df.drop(["Salary"], axis=1) # bağımsız değişkenler

# * linear regression ile rmse
lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))

y.mean()


# * cart ile rmse
cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# * cart hiperparametre optimizasyonu
# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))

