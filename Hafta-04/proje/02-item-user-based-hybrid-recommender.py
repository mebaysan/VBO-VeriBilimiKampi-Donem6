import pandas as pd

raw_movie = pd.read_csv('Datasets/movie_lens_dataset/movie.csv')
raw_rating = pd.read_csv('Datasets/movie_lens_dataset/rating.csv')

movie_df = raw_movie.copy()
rating_df = raw_rating.copy()

#############################################
######### Item-Based Recommendation #########
#############################################
def get_user_movie_df(movies,ratings):
    # movies ve ratings dataframe'lerini birleştirir. Item-Based Filtering için istediğimiz veri yapısı haline getirir. Aynı zamanda User-Based filtering de bu veri yapısını kullandığından işimizi kolaylaştıracak
    df = movie_df.merge(rating_df, how='left', on='movieId')
    comment_counts = pd.DataFrame(df['title'].value_counts()) # hangi filme (title) kaç adet değerlendirme (rating) yapılmış
    common_movies = comment_counts[comment_counts['title'] > 1000] # 1000'den fazla değerlendirmesi olan filmlerin listesi
    common_movies = df[df['title'].isin(common_movies.index)] # 1000'den fazla değerlendirmesi olan filmleri seçmek
    user_movie_df = pd.pivot(data=common_movies,index='userId',columns='movieId',values='rating') # Satırlara UserId, sütunlara MovieId ve kesişimlere ratingler gelecek şekilde veri yapısını oluşturuyoruz.
    return user_movie_df

def check_films_by_name(name,movies_df):
    # gelen dataframe'in title sütununda `name` değeri geçiyor mu diye bakacak, geçiyorsa bir listeye atıp döndürecek
    # bu fonksiyon sayesinde geliştirme aşamasında bir str değere göre filmleri arayabiliyoruz
    return movies_df[movies_df['title'].str.contains(name,na=False)]

def get_movie_name_by_id(movie_id,movies_df):
    # id'si verilen filmin adını filmlerin tutulduğu dataframe'de (movies_df) arar ve döndürür
    return movies_df[movies_df['movieId'] == movie_id]['title'].values[0]

def get_recommended_films_item_based(movie_id,user_movie_df,movies_df):
    target_movie = user_movie_df[movie_id] # hedef filmimizi seçiyoruz. user_movie_df'in kolonları her bir movie'yi temsil etmekte (benim örneğimde title değil movieId kullandım)
    recommended_movies = user_movie_df.corrwith(target_movie).sort_values(ascending=False) # ilgili filmin 
    recommended_movies = recommended_movies.iloc[1:6] # 0. index 1.0 korelasyon ile filmin kendisi olduğundan bir sonraki yüksek korelasyonlu değer ile başlıyoruz
    return [get_movie_name_by_id(index,movies_df) for index in recommended_movies.index] # her index bir id'e denk gelir. Döndüğü ID'i yukarıdaki `get_movie_name_by_id` fonksiyonuna gönderir ve isimleri liste halinde döner

check_films_by_name('Lord',movie_df) # içinde Lord geçen filmler
get_movie_name_by_id(7153,movie_df) # 7153 id'li filmin adı 

user_movie_df = get_user_movie_df(movie_df,rating_df)
get_recommended_films_item_based(7153,user_movie_df,movie_df) # 7153 id'li filmi izleyenlere önerilebilecek filmler


#############################################
######### User-Based Recommendation #########
#############################################
# *hedef kullanıcımızın izlediği ve puanladığı filmleri yakalamamız gerek
def get_user_movies(user_movie_df,user_id=None):
    # hedef user'ın izlediği filmleri verir
    if user_id == None: # eğer belirli bir user yoksa random seç
        user_id = int(pd.Series(user_movie_df.index).sample(1,random_state=6).values)
    target_user_df = user_movie_df[user_movie_df.index == user_id] # tavsiye yapacağımız kullanıcının izlediği filmler
    target_user_watched_movies_list = target_user_df.columns[target_user_df.notna().any()].tolist() # hedefimizdeki kullanıcının izlediği filmler, kesişimlerde NaN olmayanlar
    # user_movie_df.loc[user_movie_df.index==random_user,user_movie_df.columns==104] # bu kullanıcının 104 id'li filme kaç puan verdiği
    return user_id, target_user_df, target_user_watched_movies_list

target_user_id, target_user_df, target_user_watched_movies_list = get_user_movies(user_movie_df)

# *hedef kullanıcımızın izlediği filmleri yakaladığımıza göre artık bu filmleri izleyenleri yakalamamız gerekiyor
def get_same_users_with_target_user(target_user_watched_movies_list,user_movie_df,ratio=0.6):
    # hedef kullanıcımız ile aynı filmlerin en az %ratio kadarını izlemiş kullanıcıların listesini döndürür
    target_user_watched_movies_df = user_movie_df[target_user_watched_movies_list] # hedef kullanıcının izlediği filmleri user-movie matrisinden (df) çekiyoruz
    users_movie_count = target_user_watched_movies_df.T.notnull().sum() # hedefimizdeki user ile en az 1 aynı filmi izlemiş kişiler. notnull kullanarak aslında binary encode ettik burada. notnull'ları saydık ve topladık
    users_movie_count = users_movie_count.reset_index() # hangi kullanıcı kaç film izledi
    users_movie_count.columns = ["userId", "movie_count"] # toplulaştırma işleminden sonra oluşan isimlendirme problemini çözüyoruz
    users_same_movies = users_movie_count[users_movie_count["movie_count"] > len(target_user_watched_movies_list) * ratio]["userId"] # hedefimizdeki user'ın izlediği filmlerin en az %ratio kadarını izlemiş kullanıcıları seçiyoruz
    return target_user_watched_movies_df, users_same_movies

target_user_watched_movies_df , users_same_movies = get_same_users_with_target_user(target_user_watched_movies_list,user_movie_df)

# *hedef kullanıcımızın izlediği filmler ve onunla aynı filmleri izleyenler elimizde. Bunları birleştirmemiz gerekiyor ve korelasyon dataframe'lerini oluşturmamız gerekiyor
def get_corr_df_watched_movies_with_same_users(target_user_watched_movies_df,target_user_watched_movies_list,users_same_movies):
    # hedef kullanıcımız ve onunla aynı filmi izlemiş (değerlendirmiş) kullanıcıların verilerini birleştireceğiz
    final_df = pd.concat([
                      target_user_watched_movies_df[target_user_watched_movies_df.index.isin(users_same_movies)],
                      target_user_df[target_user_watched_movies_list]
                      ])
    # kullanıcılar arasında korelasyon verisini oluşturuyoruz. Her kullanıcının birbiri ile olan korelasyonunu verecek (muhtemel her çift)
    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()
    return corr_df

corr_df = get_corr_df_watched_movies_with_same_users(target_user_watched_movies_df,target_user_watched_movies_list,users_same_movies)

# *korelasyon dataframe'imizi hazırladık. Şimdi belirli bir orandan yüksek korelasyonu olan kullanıcıları getirelim
def get_top_users(corr_df,target_user_id,rating_df,ratio=0.65):
    # user_id_1 sütunu hedef userın id'si ile eşit olan satırlardan user_id_2 ve corr seçiyoruz. 
    # Bu şu demek, hedef user'ım ile user2'nin (filmleri izlemiş diğer kullanıcılar) korelasyonu
    top_users = corr_df[(corr_df['user_id_1'] == target_user_id) & (corr_df['corr'] >= ratio)][['user_id_2','corr']].reset_index(drop=True)
    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    top_users_ratings = top_users.merge(rating_df[["userId", "movieId", "rating"]], how='inner') # aynı filmleri izlemiş ve korelasyonu ratio'dan büyük kullanıcıların ratinglerini alıyoruz
    top_users_ratings = top_users_ratings[top_users_ratings["userId"] != target_user_id]
    return top_users_ratings

top_users_ratings = get_top_users(corr_df,target_user_id,rating_df)

# *korelasyonu yüksek olan kullanıcıları bulduk. Weighted average oluşturmamız lazım
def calculate_weighted_average(top_users_ratings):
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating'] # ölçüm için ağırlıklandırıyoruz
    top_users_ratings = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"}) # ölçüm birimi oluşturuyoruz
    top_users_ratings = top_users_ratings.reset_index()
    return top_users_ratings

top_users_ratings = calculate_weighted_average(top_users_ratings)

def recommend_user_based(top_users_ratings,movie_df,ratio=3.0):
    # ağırlıklı rating değeri, ratio değişkeninden yüksek olan filmleri getir
    movies_to_be_recommend = top_users_ratings[top_users_ratings["weighted_rating"] > ratio].sort_values("weighted_rating", ascending=False)
    movies_to_be_recommend = movies_to_be_recommend.merge(movie_df[["movieId", "title"]])
    return [entity for entity in movies_to_be_recommend.iloc[0:5]['title']] # ilk 5 filmi listeye ata (öner)

recommended_movies = recommend_user_based(top_users_ratings,movie_df)


#############################################
#########   Hybrid Recommendation   #########
#############################################
def golden_hybrid_recommendation(target_user_id,movies_df,ratings_df):
    user_movie_df = get_user_movie_df(movies_df,ratings_df)
    target_user_id, target_user_df, target_user_watched_movies_list = get_user_movies(user_movie_df)
    target_user_watched_movies_df , users_same_movies = get_same_users_with_target_user(target_user_watched_movies_list,user_movie_df)
    corr_df = get_corr_df_watched_movies_with_same_users(target_user_watched_movies_df,target_user_watched_movies_list,users_same_movies)
    top_users_ratings = get_top_users(corr_df,target_user_id,rating_df)
    top_users_ratings = calculate_weighted_average(top_users_ratings)
    user_based_recommended_movies = recommend_user_based(top_users_ratings,movie_df)
    last_rated_movie = ratings_df[ratings_df['userId'] == target_user_id].sort_values(by=['timestamp','rating'],ascending=[False,False]).iloc[0]['movieId']
    item_based_recommended_movies = get_recommended_films_item_based(last_rated_movie,user_movie_df,movies_df)
    recommended_movies = user_based_recommended_movies + item_based_recommended_movies
    return recommended_movies

target_user_id = 15840
golden_hybrid_recommendation(target_user_id,movie_df,rating_df)