# Yapay Zeka Destekli Mantar Zehirlilik Tespit Sistemi

## 1. Proje Özeti
Bu projede UCI Mushroom Dataset kullanılarak mantarların yenilebilir (0) veya zehirli (1) olup olmadığını tahmin eden bir ikili sınıflandırma sistemi geliştirildi. Amaç, özellikle zehirli mantarı yenilebilir olarak gösterme hatasını azaltmak ve bunu kullanıcı dostu bir Streamlit arayüzü ile sunmaktı.

## 2. Proje Teklifi ile Uyum Kontrolü
Proje mevcut haliyle teklifin ana maddeleriyle uyumludur:

- İkili sınıflandırma problemi olarak kurgulandı.
- Veri seti kategorik özelliklerden oluşan UCI Mushroom Dataset olarak kullanıldı.
- Ön işleme, model eğitimi, değerlendirme ve tahmin akışı oluşturuldu.
- Birden fazla model karşılaştırıldı.
- Yanlış negatifleri azaltma ve recall metriğini önceliklendirme yaklaşımı uygulandı.
- Son kullanıcı için Streamlit tabanlı bir arayüz hazırlandı.
- Model sonuçları ve karşılaştırma çıktıları dosyalara kaydedildi.

## 3. Basit Adım Adım Yapılanlar
1. Veri seti `data/mushrooms.csv` olarak projeye yerleştirildi.
2. Ham veri ön işleme adımından geçirildi.
3. Hedef sütun `e -> 0` ve `p -> 1` olacak şekilde dönüştürüldü.
4. Kategorik değişkenler One-Hot Encoding ile modele uygun hale getirildi.
5. Birden fazla model eğitildi ve karşılaştırıldı.
6. Nihai model seçiminde yanlış negatif sayısı önceliklendirildi.
7. Eğitimli model `models/best_model.joblib` olarak kaydedildi.
8. Değerlendirme sonuçları `results/` klasörüne yazıldı.
9. Streamlit arayüzü ile kullanıcıya tahmin ve analiz ekranı sunuldu.
10. Arayüzde Türkçe etiketler, açıklamalar ve özellik önem görselleştirmeleri düzenlendi.

## 4. Teknik Yapı
### Kullanılan Teknolojiler
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib
- Seaborn
- Plotly

### Proje Dosyaları
- `src/preprocessing.py` veri temizleme ve hazırlama
- `src/model.py` model eğitimi, karşılaştırma ve kayıt
- `src/predict.py` tekli ve toplu tahmin
- `scripts/train.py` eğitim komut satırı aracı
- `app/streamlit_app.py` kullanıcı arayüzü
- `models/best_model.joblib` kaydedilmiş eğitimli model
- `results/leaderboard.csv` model karşılaştırma tablosu
- `results/metrics.json` seçilen model metrikleri
- `results/confusion_matrix.png` confusion matrix görseli
- `notebooks/EDA.ipynb` keşifsel veri analizi
- `notebooks/model_training.ipynb` model denemeleri ve notlar

## 5. Modelleme Süreci
- Veri seti %80 eğitim ve %20 test olarak ayrıldı.
- Kategorik özellikler OneHotEncoder ile sayısallaştırıldı.
- Logistic Regression, Decision Tree, Random Forest ve Gradient Boosting modelleri karşılaştırıldı.
- Uygun ortam varsa XGBoost da destekleniyor.
- Son model seçimi yapılırken önce False Negative sayısı, sonra Recall, sonra F1 skoru dikkate alındı.

## 6. Arayüz Özeti
Streamlit arayüzünde iki ana bölüm vardır:
- Tahmin sekmesi: Kullanıcı mantar özelliklerini seçer ve anlık sonuç alır.
- Model Analizi sekmesi: Model metriği, özellik önemi ve koku odaklı analizler gösterilir.

Arayüzün amacı teknik ayrıntıyı kullanıcıdan gizleyip sonucu sade ve anlaşılır biçimde sunmaktır.

## 7. Değerlendirme Kriterlerine Göre Durum
### Fonksiyonellik
- Veri yükleme çalışıyor.
- Ön işleme çalışıyor.
- Model eğitimi çalışıyor.
- Tahmin fonksiyonu çalışıyor.
- Streamlit arayüzü çalışıyor.

### Kod Kalitesi
- Kod modüler şekilde ayrıldı.
- Ön işleme, modelleme ve tahmin ayrı dosyalarda tutuldu.
- Eğitim akışı komut satırından çalıştırılabiliyor.

### Arayüz
- Türkçe ve sade bir kullanıcı akışı kuruldu.
- Sonuçlar renkli kartlarla veriliyor.
- Özellik önemleri ve analiz grafikleri gösteriliyor.

## 9. Sonuç
Proje, teklif edilen temel kapsamı karşılıyor: veri ön işleme, model karşılaştırma, recall odaklı seçim, Streamlit arayüzü ve eğitimli model çıktısı hazır. Bu nedenle proje teslimi için ana teknik yapı tamamlanmış durumdadır.
