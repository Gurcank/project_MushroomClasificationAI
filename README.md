# Yapay Zeka Destekli Mantar Zehirlilik Tahmin Modeli
Bulut Bilişim ve Yapay Zeka dersim için UCI Mushroom Dataset kullanarak , mantarların "Yenilebilir (0)" veya "Zehirli (1)" olduğunu tahmin eden bir model yaptım.

## Proje Amacı
Mantarın morfolojik özelliklerinden zehirlilik tespiti yapmak

## Projede Gerçekleştirilen Adımlar
1-Veri setini data/mushrooms.csv olarak projeye yerleştirdim.

2-Ham veriyi preprocecsing işleminden geçirdim.

3-Hedef sutun e -> 0 ve p -> 1 olacak şekilde dönüştürdüm.

4-Kategorik değişkenler One-Hot Encoding ile modele uygun hale getirdim.

5-Birden fazla modeli eğitip karşılaştırdım.

6-En iyi modeli seçerken yanlış negatif sayısına önem verdim.

7-Eğitimli modeli models/best_model.joblib olarak kaydettim.

8-Değerlendirme sonuçlarını results/ klasörüne yazdım.

9-Streamlit arayüzü ile kullanıcıya tahmin ve analiz ekranı oluşturdum.


## Kullanılan Teknolojiler
-Python
-Pandas
-NumPy
-Scikit-learn
-Streamlit
-Joblib
-Matplotlib
-Seaborn
-Plotly

## Proje Dosyaları
-src/preprocessing.py             Preprocessing işlemlerinin gerçekleştiği dosya

-src/model.py                     Logistic Regression, Decision Tree, Random Forest, Gradient Boosting gibi modelleri kurup deneyip en iyisini seçer.

-src/predict.py                   Seçilen modeli kullanarak tek ve toplu kayıt için tahmin üretir.

-scripts/train.py                 Eğitim sürecini başlatır modeli eğitir ve en iyisini kaydeder.

-app/streamlit_app.py             UI/UX kısmı

-models/best_model.joblib         Kaydedilen en iyi modeldir.Uygulama tahmin yaparken bunu kullanır.

-results/leaderboard.csv          Denenen modellerin karşılaştırmalı tablosudur.

-results/metrics.json             Seçilen modelin metriklerini tutar.

-notebooks/EDA.ipynb              Keşifsel veri analizi burda yapılır.

-notebooks/model_training.ipynb   Modeller arasından en iyisini seçerkenki sürece ait notlar.

## Modelleme Süreci
-Veri setini %80 eğitim ve %20 test olarak ayırdım.

-Kategorik özellikleri OneHotEncoder ile sayısallaştırıldım.

-Logistic Regression, Decision Tree, Random Forest ve Gradient Boosting modellerini karşılaştırıldım.

-False Negative sayısı, sonra Recall, sonra F1 skorunu dikkate alarak en iyi modeli seçtim.

## UI/UX Kısmı
Streamlit kullanarak 2 sekme oluşturdum:

-Tahmin sekmesi: Kullanıcının mantar özelliklerini seçip sonucunu görebildiği sekme.

-Model Analizi sekmesi: Modelin özelliklerini ve modelle ilgili grafiklerin yer aldığı bilgilendirme sekmesi.
