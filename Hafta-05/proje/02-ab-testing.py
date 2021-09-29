import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu


def baysan_independent_t_test(var1, var2):
    """2 değişkene standart bağımsız 2 örneklem t testi adımlarını ve kontrollerini uygulayarak H0'ın red edilip edilemeyeceğini döndürür

    Args:
        var1 (pd.Series): Değişken1
        var2 (pd.Series): Değişken2
    """
    def check_normality(var1, var2):
        """2 değişkenin normalliğini kontrol eder

        Args:
            var1 (pd.Series): normallik varsayımı yapılmak istenen değişken
            var2 (pd.Series): normallik varsayımı yapılmak istenen değişken

        Returns:
            bool: 2 değişkende normal dağılıyorsa True yoksa False döner.
                  H0 hipotezi değişken normal dağılıyordur. p değeri 0.05'ten büyük çıkarsa H0 red edilemezdir yani bizim için True normal dağılıyor demektir.
        """
        test_stat1, p1 = shapiro(var1)
        test_stat2, p2 = shapiro(var2)
        return True if p1 > 0.05 and p2 > 0.05 else False

    def check_variance_homogenity(var1, var2):
        """2 değişken arasında varyans homojenliğini kontrol eder.

        Args:
            var1 (pd.Series): Varyans homojenliği için 1. değişken
            var2 (pd.Series): Varyans homojenliği için 2. değişken

        Returns:
            bool: Varyans homojenliği sağlanıyorsa True, sağlanmıyorsa False döner.
                  H0 hipotezi: değişkenler arasında varyans homojenliği vardır. p değeri 0.05'ten büyük ise H0 red edilemez yani varyans homojenliği sağlanıyordur ki bu bizim için True demektir.
        """
        test_stat, p = levene(var1, var2)
        return True if p > 0.05 else False
    
    def get_hypothesis_result(p):
        """P değerine göre H0 hipotez sonucu verir

        Args:
            p (float): Test sonucu elde edilen p-value

        Returns:
            str: p değeri 0.05'ten büyük ise "H0 Red Edilemez", değil ise "H0 Red Edilebilir" döner
        """
        return f"H0 Red Edilemez ------- p-value: {round(p,5)}" if p > 0.05 else f"H0 Red Edilebilir ------- p-value: {round(p,5)}"


    if check_normality(var1,var2) and check_variance_homogenity(var1,var2): # * normallik ve varyans homojenliği sağlanıyorsa bağımsız 2 önreklem t test
        test_stat, p = ttest_ind(var1,var2,equal_var=True)
        return get_hypothesis_result(p)
    elif not check_normality(var1, var2): # * normallik sağlanmıyorsa direkt mannwhithneyu
        test_stat, p = mannwhitneyu(var1,var2)
        return get_hypothesis_result(p)
    elif check_normality(var1,var2) and not check_variance_homogenity(var1,var2):
        test_stat, p = ttest_ind(var1,var2,equal_var=False)
        return get_hypothesis_result(p)
    else:
        raise RuntimeError("Bilinmeyen bir hata oluştu!")


control_df = pd.read_excel('Datasets/ab_testing.xlsx', sheet_name='Control Group')
test_df = pd.read_excel('Datasets/ab_testing.xlsx', sheet_name='Test Group')

control_df.head()
test_df.head()

# * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# * H0 HİPOTEZİ YOKLUK HİPOTEZİDİR
# * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

########################################
# * Uygulama 1: Kontrol ve Test grupları arasındaki görüntülenme sayıları arasında istatistiksel olarak bir anlamlılık var mıdır?
########################################
# H0: Kontrol ve Test grupları arasındaki görüntülenme sayıları arasında istatistiksel olarak bir anlamlılık yoktur
# H1: Kontrol ve Test grupları arasındaki görüntülenme sayıları arasında istatistiksel olarak bir anlamlılık vardır
baysan_independent_t_test(
    control_df['Impression'],
    test_df['Impression']
    )



########################################
# * Uygulama 2: Kontrol ve Test grupları arasındaki tıklama arasında istatistiksel olarak bir anlamlılık var mıdır?
########################################
# H0: Kontrol ve Test grupları arasındaki tıklanma sayıları arasında istatistiksel olarak bir anlamlılık yoktur
# H1: Kontrol ve Test grupları arasındaki tıklanma sayıları arasında istatistiksel olarak bir anlamlılık vardır
baysan_independent_t_test(
    control_df['Click'],
    test_df['Click']
    )



########################################
# * Uygulama 3: Kontrol ve Test grupları arasındaki kazançlar (earning) arasında istatistiksel olarak bir anlamlılık var mıdır?
########################################
# H0: Kontrol ve Test grupları arasındaki kazançlar arasında istatistiksel olarak bir anlamlılık yoktur
# H1: Kontrol ve Test grupları arasındaki kazançlar arasında istatistiksel olarak bir anlamlılık vardır
baysan_independent_t_test(
    control_df['Earning'],
    test_df['Earning']
    )


########################################
# * Uygulama 4: Kontrol ve Test grupları arasındaki satın alımlar arasında istatistiksel olarak bir anlamlılık var mıdır?
########################################
# H0: Kontrol ve Test grupları arasındaki satın alımlar arasında istatistiksel olarak bir anlamlılık yoktur
# H1: Kontrol ve Test grupları arasındaki satın alımlar arasında istatistiksel olarak bir anlamlılık vardır
baysan_independent_t_test(
    control_df['Purchase'],
    test_df['Purchase']
    )

# * Sonuç: Karşılaştırılan 2 grup arasında (%95 güvenilirlik ile):
# *         - "Average Bidding" sistemi görüntülenme, tıklanma ve kazançları artırmamıştır
# *         - Fakat satın alım işlem adedini artırmıştır. Yeni Bidding sistemi ile daha fazla transaction gerçekleşmiştir.
control_df['Purchase'].mean()
test_df['Purchase'].mean()
# * Bu çalışma sonucu bizim önerimiz bir süre daha bu sistemin gözlemlenmesi yönünde olacaktır.