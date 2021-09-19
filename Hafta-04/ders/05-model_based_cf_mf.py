#############################
# Model Based Collaborative Filtering: Matrix Factorization
#############################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

# pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)



movie = pd.read_csv('Datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()


movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]


sample_df = df[df.movieId.isin(movie_ids)] # gözlemleyebilmek ve hızdan tasarruf etmek için veriyi azaltıyoruz
sample_df.shape
sample_df.head()

# user movie df'inin oluşturulması
user_movie_df = sample_df.pivot_table(index=["userId"], columns=["title"], values="rating") # Item-Based filtering de kullandığımız veri yapısını kullanıyoruz
user_movie_df.shape
user_movie_df.head()

reader = Reader(rating_scale=(1, 5)) # skorların kaç ile kaç arasında olduğunu belirtiyoruz (rating limitleri)
data = Dataset.load_from_df(sample_df[['userId', 'movieId', 'rating']], reader) # veriyi kütüphanenin (surprise) istediği formata sokuyoruz

##############################
# Adım 2: Modelleme
##############################

trainset, testset = train_test_split(data, test_size=.25) # veriyi train ve test olarak 2'ye ayırdık. %25'i test olarak
svd_model = SVD() # matrix factorization için bir model oluşturuyoruz
svd_model.fit(trainset) # modeli train seti ile eğitiyoruz
predictions = svd_model.test(testset) # modeli test verileri ile test ediyoruz

accuracy.rmse(predictions) # root mean squared error hesaplıyoruz. 

cross_validate(svd_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True) 
# RMSE -> Root Mean Squarred Error, gerçek değerler ile tahmin edilen değerlerin farklarının karelerinin ortalamasının kare köküküdür. Test setindeki başarımızdır diyebiliriz
# MAE -> Mean Absolute Error, gerçek değerler ile tahmin edilen değerlerin farklarının mutlak değerinin ortalamasıdır
# veri setini 5 parçaya böl, 4 parçasıyla model kur diğer 1 parçasıyla test et. Daha sonra 1 parçasıyla model kur, diğer 4 parçasıyla test et diyoruz.

# user id 1 için, blade runner (541 id) tahmini
# uid -> user id, iid -> item id
svd_model.predict(uid=1.0, iid=541, verbose=True)
# >>> Prediction(uid=1.0, iid=541, r_ui=None, est=4.192434293734314, details={'was_impossible': False}) ||| est parametresi tahmin edilen değeri göstermektedir. Gerçek değeri kontrol edersek ise userın 4.0 rating verdiğini göreceğiz

svd_model.predict(uid=1.0, iid=356, verbose=True) # 1 id'li user'a 356 id'li filmi önerirsek beklenilen beğenmesi nedir? (est)


##############################
# Adım 3: Model Tuning
##############################

# GridSearchCV
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]} # hiper parametreler
# n_epoch -> kaç defa ağırlık güncellemesi (p ve q) yapacağız
# lr_all -> learning rate, epochlarda ağırlık setlerinin değerlerini hangi hızda değiştireceğiz

gs = GridSearchCV(SVD,
                  param_grid, # hiper parametreler
                  measures=['rmse', 'mae'], # hata ölçüm metrikleri
                  cv=3, # kaç kere cross validation yapılacak
                  n_jobs=-1, # tüm işlemcileri çalıştır
                  joblib_verbose=True)

gs.fit(data)
gs.best_score['rmse'] 
gs.best_params['rmse'] # model için en iyi hiper parametre değerlerini buluyoruz


##############################
# Adım 4: Final Model ve Tahmin
##############################
# en iyi hiper parametre değerleri ile yeni bir final model kurmamız gerekiyor
svd_model = SVD(**gs.best_params['rmse'])

data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)
svd_model.predict(uid=1.0, iid=356, verbose=True)



# http://sifter.org/~simon/journal/20061211.html
# https://www.freecodecamp.org/news/singular-value-decomposition-vs-matrix-factorization-in-recommender-systems-b1e99bc73599/
# https://surprise.readthedocs.io/en/stable/getting_started.html