# Yapay Zeka Destekli Mantar Zehirlilik Tahmin Modeli
Bulut Bilişim ve Yapay Zeka dersim için UCI Mushroom Dataset kullanarak , mantarların "Yenilebilir (0)" veya "Zehirli (1)" olduğunu tahmin eden bir model yaptım.

## Proje Amacı
Mantarın morfolojik özelliklerinden zehirlilik tespiti yapmak

## Projede Gerçekleştirilen Adımlar
1-Veri seti data/mushrooms.csv olarak projeye yerleştirildi.
2-Ham veri ön işleme adımından geçirildi.
3-Hedef sütun e -> 0 ve p -> 1 olacak şekilde dönüştürüldü.
4-Kategorik değişkenler One-Hot Encoding ile modele uygun hale getirildi.
5-Birden fazla model eğitildi ve karşılaştırıldı.
6-Nihai model seçiminde yanlış negatif sayısı önceliklendirildi.
7-Eğitimli model models/best_model.joblib olarak kaydedildi.
8-Değerlendirme sonuçları results/ klasörüne yazıldı.
9-Streamlit arayüzü ile kullanıcıya tahmin ve analiz ekranı sunuldu.
10-Arayüzde Türkçe etiketler, açıklamalar ve özellik önem görselleştirmeleri düzenlendi.

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
