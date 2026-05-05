# Yapay Zeka Destekli Mantar Zehirlilik Tespit Sistemi

Bu proje, UCI Mushroom Dataset kullanarak mantarların **yenilebilir (0)** veya **zehirli (1)** olduğunu tahmin eden bir binary classification sistemidir.

## Proje Amacı
- Mantarın morfolojik özelliklerinden zehirlilik tespiti yapmak
- Yanlış negatifleri (zehirli mantarı yenilebilir gösterme) azaltmak
- Modeli basit bir Streamlit arayüzü ile son kullanıcıya sunmak

## Teknoloji Yığını
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib, Seaborn, Plotly

## Klasör Yapısı
- `data/` ham veri seti
- `notebooks/` EDA ve model deneme not defterleri
- `src/` ön işleme, modelleme ve tahmin modülleri
- `scripts/` eğitim komutları
- `app/` Streamlit arayüzü
- `models/` kaydedilmiş model artifact'i
- `results/` metrikler ve görseller

## Not Defterleri
- `notebooks/EDA.ipynb` veri keşfi ve ilk analizler
- `notebooks/model_training.ipynb` model karşılaştırma ve seçim denemeleri

## Kurulum
```bash
python -m pip install -r requirements.txt
```

## Veri Seti
Kaggle UCI Mushroom Dataset dosyasını `data/mushrooms.csv` olarak yerleştir.

## Model Eğitimi
```bash
python -m scripts.train --data data/mushrooms.csv --output models/best_model.joblib
```

İstersen `?` ile işaretli kayıtları silmek için:
```bash
python -m scripts.train --data data/mushrooms.csv --output models/best_model.joblib --drop-unknown-rows
```

Egitim sonrasi `results/` altinda su dosyalar olusur:
- `leaderboard.csv` model karsilastirma tablosu
- `metrics.json` secilen model metrikleri
- `confusion_matrix.png` confusion matrix gorseli

## Arayüzü Çalıştırma
```bash
python -m streamlit run app/streamlit_app.py
```

Arayuz ozellikleri:
- Tahmin sekmesi: 22 ozellik girisi, olasilik ve risk odakli sonuc karti
- Model Analizi sekmesi: metrik tablosu ve ozellik onem bar grafigi

## Notlar
- Nihai model seçiminde accuracy yerine **false negative minimizasyonu** ve **recall** önceliklidir.
- Zehirli mantarı yenilebilir olarak etiketlemek kritik hata kabul edilir.

## Teslim İçin Önerilen Çıktılar
- GitHub repo
- README
- Ekran görüntüleri
- Demo video veya kısa rapor
- Eğitimli model dosyası
