dsmlbc5 reposundaki fonksiyonlara docstring yazınız ve PR yapınız.


1. https://github.com/mvahit/dsmlbc5 adresindeki repoyu fork ediniz.
2. Fork ettikten sonra kendi reponuz haline gelen repoda yer alan yeşil renkteki CODE bölümüne tıklayınız ve HTTPS bölümünde yer alan linki kopyalayınız.
3. Komut satırından masaüstüne geliniz ve repoyu clone’layınız (git clone “forklanan_repo_linki”) (bir önceki maddedeki link)
4. Klonlanan reponun klasörüne giriniz. (cd dsmlbc5)
5. Kendi adınızdan oluşan yeni bir branch oluşturunuz. (git checkout -b ad_soyad)
6. eda.py dosyasını açınız ve fonksiyon ekleyiniz.
7. add ve commit işlemlerini yapınız (git add eda.py) (git commit -m “fonksiyon eklendi”)
8. master branch'e geçiniz. (git checkout master)   
9. İsim soy isminizden oluşan branch’inizi push ediniz. (git push origin ad_soyad) (GitHub kullanıcı adı ve şifrenizi isteyecek bunları giriniz.)
10. Github üzerinden fork ettiğiniz reponuza geliniz.
11. “pull requests” bölümüne tıklayınız. "compare & pull request" bölümüne tıklayın.
12. base ve compare bölümlerini kontrol ediniz. sol tarafta master sağ tarafta sizin branch'iniz olmalı. Değilse düzeltiniz.
13. Notunuzu yazınız ve create pull request'e tıklayınız.


Önemli not 1!
Yeni açtığınız branch'i github'a push etmeden önce git add, git commit yapmayı unutmayın!!!

Önemli not 2!
Eğer birkaç defa PR yaptıysanız 11. maddedeki gibi pr'ınızı göremeyebilirsiniz.
Bu durumda localden push ettikten sonra code bölümünde branches'lara tıklayıp buradan istediğiniz branch için tekrar PR başlatınız.

Önemli not 3!
Bir branch merge olduktan sonra silinir!
Tekrar PR yapmak istediğinizde yeni bir branch oluşturunuz.
Hatalardan kaçınmak için de yeni bir isimlendirme yapabilirsiniz.

Önemli not 4!
Aşağıdaki hatayı alırsanız şu kod ile pull yapınız: git pull --rebase
"updates were rejected because the tip of your current branch is behind"
