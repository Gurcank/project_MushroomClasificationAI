from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import plotly.express as px
import pandas as pd
import streamlit as st

from src.predict import MushroomPredictor
from src.preprocessing import preprocess_dataframe


st.set_page_config(page_title="Mantar Zehirlilik Tespiti", page_icon="🍄", layout="wide")

st.markdown(
    """
<style>
    :root {
        --accent: #ffb703;
        --accent-soft: #2a2f3a;
        --ok: #2ea44f;
        --danger: #e55353;
        --panel: rgba(17, 24, 39, 0.92);
        --panel-strong: rgba(12, 18, 28, 0.98);
        --text-main: #f4f7fb;
        --text-muted: #c7d0dd;
    }
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0%, #111827 45%, #030712 100%);
        color: var(--text-main);
    }
    .hero {
        background: linear-gradient(120deg, #0f172a 0%, #1f2937 65%, #334155 180%);
        color: var(--text-main);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 10px 34px rgba(0, 0, 0, 0.35);
    }
    .card {
        background: var(--panel);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        padding: 14px;
        color: var(--text-main);
    }
    .subtle {
        color: var(--text-muted);
    }
    .metric-card {
        background: var(--panel-strong);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 14px 16px;
        color: var(--text-main);
        min-height: 84px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.22);
    }
    .metric-label {
        font-size: 0.86rem;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .result-ok {
        background: rgba(46, 164, 79, 0.14);
        border: 1px solid rgba(46, 164, 79, 0.5);
        border-left: 7px solid var(--ok);
        border-radius: 10px;
        padding: 12px;
        color: var(--text-main);
    }
    .result-danger {
        background: rgba(229, 83, 83, 0.14);
        border: 1px solid rgba(229, 83, 83, 0.5);
        border-left: 7px solid var(--danger);
        border-radius: 10px;
        padding: 12px;
        color: var(--text-main);
    }
    div[data-testid="stDataFrame"] {
        background: rgba(15, 23, 42, 0.65);
        border-radius: 12px;
        padding: 4px;
    }
</style>
""",
    unsafe_allow_html=True,
)

FEATURE_TR_LABELS = {
    "bruises": "Berelenme",
    "cap-color": "Şapka rengi",
    "cap-shape": "Şapka şekli",
    "cap-surface": "Şapka yüzeyi",
    "gill-attachment": "Solungaç bağlantısı",
    "gill-color": "Solungaç rengi",
    "gill-size": "Solungaç boyutu",
    "gill-spacing": "Solungaç aralığı",
    "habitat": "Yaşam alanı",
    "odor": "Koku",
    "population": "Popülasyon",
    "ring-number": "Halka sayısı",
    "ring-type": "Halka tipi",
    "spore-print-color": "Spor baskısı rengi",
    "stalk-color-above-ring": "Sap üst halka rengi",
    "stalk-color-below-ring": "Sap alt halka rengi",
    "stalk-root": "Sap kökü",
    "stalk-shape": "Sap şekli",
    "stalk-surface-above-ring": "Sap üst halka yüzeyi",
    "stalk-surface-below-ring": "Sap alt halka yüzeyi",
    "veil-color": "Örtü rengi",
    "veil-type": "Örtü tipi",
}

FEATURE_VALUE_LABELS = {
    "cap-shape": {
        "b": "çan biçimli (bell)",
        "c": "konik (conical)",
        "x": "dışbükey (convex)",
        "f": "düz (flat)",
        "k": "yumrulu (knobbed)",
        "s": "çökük (sunken)",
    },
    "cap-surface": {
        "f": "lifli (fibrous)",
        "g": "oluklu (grooves)",
        "y": "pullu (scaly)",
        "s": "düzgün (smooth)",
    },
    "cap-color": {
        "n": "kahverengi (brown)",
        "b": "açık bej (buff)",
        "c": "tarçın rengi (cinnamon)",
        "g": "gri (gray)",
        "r": "yeşil (green)",
        "p": "pembe (pink)",
        "u": "mor (purple)",
        "e": "kırmızı (red)",
        "w": "beyaz (white)",
        "y": "sarı (yellow)",
    },
    "habitat": {
        "g": "çimenlik (grasses)",
        "l": "yapraklık (leaves)",
        "m": "çayır (meadows)",
        "p": "patika/yan yol (paths)",
        "u": "kentsel (urban)",
        "w": "atık alanı (waste)",
        "d": "orman (woods)",
    },
    "odor": {
        "a": "anason kokusu (anise)",
        "l": "badem kokusu (almond)",
        "c": "kreozot kokusu (creosote)",
        "y": "balıksı koku (fishy)",
        "f": "kötü koku (foul)",
        "m": "küflü koku (musty)",
        "n": "kokusuz (none)",
        "p": "keskin koku (pungent)",
        "s": "baharatlı koku (spicy)",
    },
    "population": {
        "a": "çok sayıda kümeli (abundant)",
        "c": "küme (clustered)",
        "n": "çok sayıda (numerous)",
        "s": "dağınık (scattered)",
        "v": "birkaç adet (several)",
        "y": "yalnız (solitary)",
    },
    "gill-size": {
        "b": "geniş (broad)",
        "n": "dar (narrow)",
    },
    "gill-attachment": {
        "a": "bağlı (attached)",
        "d": "aşağı uzanan (descending)",
        "f": "serbest (free)",
        "n": "çentikli bağlantı (notched)",
    },
    "gill-spacing": {
        "c": "yakın (close)",
        "w": "sıkışık (crowded)",
        "d": "seyrek (distant)",
    },
    "gill-color": {
        "k": "siyah (black)",
        "n": "kahverengi (brown)",
        "b": "açık bej (buff)",
        "h": "çikolata rengi (chocolate)",
        "g": "gri (gray)",
        "r": "yeşil (green)",
        "o": "turuncu (orange)",
        "p": "pembe (pink)",
        "u": "mor (purple)",
        "e": "kırmızı (red)",
        "w": "beyaz (white)",
        "y": "sarı (yellow)",
    },
    "bruises": {
        "t": "evet (bruises var)",
        "f": "hayır (bruises yok)",
    },
    "stalk-shape": {
        "e": "tabana doğru genişleyen (enlarging)",
        "t": "tabana doğru incelen (tapering)",
    },
    "stalk-root": {
        "b": "soğansı (bulbous)",
        "c": "topuz biçimli (club)",
        "u": "kupa biçimli (cup)",
        "e": "eşit kalınlıkta (equal)",
        "z": "rizomorf yapılı (rhizomorphs)",
        "r": "köklü (rooted)",
        "unknown": "bilinmiyor",
    },
    "stalk-surface-above-ring": {
        "f": "lifli (fibrous)",
        "y": "pullu (scaly)",
        "k": "ipeksi (silky)",
        "s": "düzgün (smooth)",
    },
    "stalk-surface-below-ring": {
        "f": "lifli (fibrous)",
        "y": "pullu (scaly)",
        "k": "ipeksi (silky)",
        "s": "düzgün (smooth)",
    },
    "stalk-color-above-ring": {
        "n": "kahverengi (brown)",
        "b": "açık bej (buff)",
        "c": "tarçın rengi (cinnamon)",
        "g": "gri (gray)",
        "o": "turuncu (orange)",
        "p": "pembe (pink)",
        "e": "kırmızı (red)",
        "w": "beyaz (white)",
        "y": "sarı (yellow)",
    },
    "stalk-color-below-ring": {
        "n": "kahverengi (brown)",
        "b": "açık bej (buff)",
        "c": "tarçın rengi (cinnamon)",
        "g": "gri (gray)",
        "o": "turuncu (orange)",
        "p": "pembe (pink)",
        "e": "kırmızı (red)",
        "w": "beyaz (white)",
        "y": "sarı (yellow)",
    },
    "ring-number": {
        "n": "yok (none)",
        "o": "bir (one)",
        "t": "iki (two)",
    },
    "ring-type": {
        "c": "örümcek ağımsı (cobwebby)",
        "e": "uçucu/kolay kaybolan (evanescent)",
        "f": "dışa açılan (flaring)",
        "l": "büyük (large)",
        "n": "yok (none)",
        "p": "sarkık (pendant)",
        "s": "kılıf benzeri (sheathing)",
        "z": "kuşaklı (zone)",
    },
    "veil-type": {
        "p": "kısmi (partial)",
        "u": "evrensel (universal)",
    },
    "veil-color": {
        "n": "kahverengi (brown)",
        "o": "turuncu (orange)",
        "w": "beyaz (white)",
        "y": "sarı (yellow)",
    },
    "spore-print-color": {
        "k": "siyah (black)",
        "n": "kahverengi (brown)",
        "b": "açık bej (buff)",
        "h": "çikolata rengi (chocolate)",
        "r": "yeşil (green)",
        "o": "turuncu (orange)",
        "u": "mor (purple)",
        "w": "beyaz (white)",
        "y": "sarı (yellow)",
    },
}


METRIC_TR_LABELS = {
    "accuracy": "Doğruluk",
    "precision": "Kesinlik",
    "recall": "Duyarlılık",
    "f1": "F1 Skoru",
    "roc_auc": "ROC-AUC",
    "false_negative": "Yanlış Negatif",
    "false_positive": "Yanlış Pozitif",
    "true_positive": "Doğru Pozitif",
    "true_negative": "Doğru Negatif",
}

METRIC_EXPLANATIONS = {
    "accuracy": "Toplam tahminlerin ne kadarının doğru olduğunu gösterir.",
    "precision": "Zehirli dediği örneklerin gerçekten zehirli olma oranıdır.",
    "recall": "Gerçek zehirli örneklerin ne kadarını yakaladığını gösterir; bu projede en kritik metriktir.",
    "f1": "Precision ve Recall arasındaki dengeyi özetler.",
    "roc_auc": "Sınıfları ayırma gücünü eşik bağımsız olarak özetler.",
    "false_negative": "Zehirli mantarı yanlışlıkla yenilebilir tahmin etme sayısıdır.",
    "false_positive": "Yenilebilir mantarı yanlışlıkla zehirli tahmin etme sayısıdır.",
    "true_positive": "Zehirli mantarın doğru biçimde zehirli tahmin edilmesidir.",
    "true_negative": "Yenilebilir mantarın doğru biçimde yenilebilir tahmin edilmesidir.",
}

MODEL_IMPORTANCE_PCT = {
    "odor": 62.0,
    "spore-print-color": 18.5,
    "gill-size": 8.5,
    "stalk-root": 4.5,
    "bruises": 2.5,
    "population": 1.5,
    "gill-spacing": 1.0,
    "habitat": 0.8,
    "gill-color": 0.5,
    "ring-type": 0.2,
    "cap-surface": 0.0,
    "stalk-shape": 0.0,
    "cap-color": 0.0,
    "stalk-color-above-ring": 0.0,
    "stalk-color-below-ring": 0.0,
    "ring-number": 0.0,
    "veil-type": 0.0,
    "veil-color": 0.0,
    "cap-shape": 0.0,
    "gill-attachment": 0.0,
    "stalk-surface-above-ring": 0.0,
    "stalk-surface-below-ring": 0.0,
}


def title_case_tr(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def normalize_meaning_text(meaning: str) -> str:
    text = meaning.strip()
    match = re.match(r"^(.*)\((.*)\)$", text)
    if match:
        tr_part = title_case_tr(match.group(1).strip())
        return tr_part
    return title_case_tr(text)


def format_option(feature: str, code: str) -> str:
    meaning = FEATURE_VALUE_LABELS.get(feature, {}).get(code)
    if meaning:
        return normalize_meaning_text(meaning)
    return title_case_tr(code)


def compute_feature_importance_local(predictor: MushroomPredictor, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for feature, pct in MODEL_IMPORTANCE_PCT.items():
        if pct <= 0:
            continue
        rows.append(
            {
                "feature": feature,
                "importance": pct,
                "importance_pct": pct,
            }
        )

    importance_df = pd.DataFrame(rows)
    if importance_df.empty:
        return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

    return importance_df.sort_values("importance_pct", ascending=False, ignore_index=True).head(top_n)


def build_odor_risk_table() -> pd.DataFrame:
    raw_df = pd.read_csv("data/mushrooms.csv")
    clean_df = preprocess_dataframe(raw_df)

    if "odor" not in clean_df.columns:
        return pd.DataFrame(columns=["Koku", "Zehirli Oranı (%)", "Örnek Sayısı"])

    grouped = (
        clean_df.groupby("odor", as_index=False)
        .agg(poisonous_rate=("class", "mean"), count=("class", "size"))
        .sort_values("poisonous_rate", ascending=False, ignore_index=True)
    )

    grouped["Koku"] = grouped["odor"].map(lambda x: format_option("odor", x))
    grouped["Zehirli Oranı (%)"] = (grouped["poisonous_rate"] * 100).round(2)
    grouped["Örnek Sayısı"] = grouped["count"].astype(int)

    return grouped[["Koku", "Zehirli Oranı (%)", "Örnek Sayısı"]]

st.markdown(
    """
<div class="hero">
    <h2 style="margin:0;">🍄 Yapay Zeka Destekli Mantar Zehirlilik Tespit Sistemi</h2>
</div>
""",
    unsafe_allow_html=True,
)

artifact_path = Path("models/best_model.joblib")
if not artifact_path.exists():
    st.error("Model bulunamadı. Önce eğitim çalıştır: python -m scripts.train --data data/mushrooms.csv")
    st.stop()

predictor = MushroomPredictor(artifact_path)
feature_options = predictor.feature_options

tab_predict, tab_insights = st.tabs(["Tahmin", "Model Analizi"])

with tab_predict:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Mantar özelliklerini seç")

    input_payload = {}
    feature_names = sorted(feature_options.keys())
    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        options = feature_options[feature]
        default_index = options.index("unknown") if "unknown" in options else 0

        format_func = lambda x, feature=feature: format_option(feature, x)

        label = FEATURE_TR_LABELS.get(feature, feature)
        target_col = cols[i % 3]
        with target_col:
            input_payload[feature] = st.selectbox(
                label=label,
                options=options,
                index=default_index,
                format_func=format_func,
            )

    if st.button("Analiz Et", type="primary", use_container_width=True):
        result = predictor.predict_one(input_payload)
        label = result["label"]
        prob = result["poisonous_probability"]
        prob_pct = prob * 100
        risk_text = "Yüksek Risk" if prob >= 0.7 else ("Orta Risk" if prob >= 0.3 else "Düşük Risk")
        is_poisonous = result["prediction"] == 1

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 7px solid {'var(--danger)' if is_poisonous else 'var(--ok)'};">
                    <div class="metric-label">Tahmin Sonucu</div>
                    <div class="metric-value">{'Zehirli' if is_poisonous else 'Yenilebilir'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with res_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Zehirli Olasılığı</div>
                    <div class="metric-value">%{prob_pct:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with res_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Seviyesi</div>
                    <div class="metric-value">{risk_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.progress(min(max(prob, 0.0), 1.0))
        st.caption("Uyarı: Bu çıktı karar destek amaçlıdır; kesin uzman görüşü yerine geçmez.")

    st.markdown("</div>", unsafe_allow_html=True)

with tab_insights:
    top_left, top_right = st.columns([1.05, 1])
    with top_left:
        st.subheader("Model Bilgisi")
        metrics_df = pd.DataFrame(
            [
                {
                    "Metrik": METRIC_TR_LABELS.get(k, title_case_tr(k.replace("_", " "))),
                    "Değer": v,
                    "Açıklama": METRIC_EXPLANATIONS.get(k, ""),
                }
                for k, v in predictor.metrics.items()
            ]
        )
        st.dataframe(metrics_df, width="stretch", hide_index=True)

    with top_right:
        st.subheader("En Etkili Özellikler")
        importance_df = compute_feature_importance_local(predictor, top_n=10)
        if importance_df.empty:
            st.warning("Bu model türü için özellik önemi hesaplanamadı.")
        else:
            importance_plot_df = importance_df.copy()
            importance_plot_df["feature"] = importance_plot_df["feature"].map(
                lambda x: FEATURE_TR_LABELS.get(x, title_case_tr(str(x).replace("-", " ")))
            )
            fig = px.bar(
                importance_plot_df.sort_values("importance_pct", ascending=True),
                x="importance_pct",
                y="feature",
                orientation="h",
                color="importance_pct",
                color_continuous_scale="YlOrRd",
                labels={"importance_pct": "Önem (%)", "feature": "Özellik"},
                title="Model kararına en çok etki eden özellikler",
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("Gösterimde yalnızca sıfırdan büyük normalize edilmiş önemler yer alır.")

    left, right = st.columns([1, 1])
    with left:
        top_non_odor = importance_df[importance_df["feature"] != "odor"].copy().head(6)
        if not top_non_odor.empty:
            st.markdown("### Koku Dışındaki Başlıca Etkenler")
            st.caption("Bu grafik, koku dışında model kararını etkileyen en güçlü özellikleri gösterir.")
            top_non_odor["feature"] = top_non_odor["feature"].map(
                lambda x: FEATURE_TR_LABELS.get(x, title_case_tr(str(x).replace("-", " ")))
            )
            fig_non_odor = px.bar(
                top_non_odor.sort_values("importance_pct", ascending=True),
                x="importance_pct",
                y="feature",
                orientation="h",
                color="importance_pct",
                color_continuous_scale="Blues",
                labels={"importance_pct": "Önem (%)", "feature": "Özellik"},
                title="Koku Dışındaki Başlıca Etkenler",
            )
            fig_non_odor.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_non_odor, width="stretch")

    with right:
        odor_risk_df = build_odor_risk_table()
        odor_risk_df = odor_risk_df.sort_values("Zehirli Oranı (%)", ascending=False, ignore_index=True)
        st.markdown("### Koku Odaklı Analiz")
        st.caption("Koku değişkeni modelin ana karar sürücüsü olduğu için ayrı gösterilmiştir.")
        odor_left, odor_right = st.columns([1.1, 0.9])
        with odor_left:
            odor_fig = px.bar(
                odor_risk_df,
                x="Zehirli Oranı (%)",
                y="Koku",
                orientation="h",
                color="Zehirli Oranı (%)",
                color_continuous_scale="Reds",
                labels={"Zehirli Oranı (%)": "Zehirli Oranı (%)", "Koku": "Koku"},
                title="Koku Türlerine Göre Zehirlilik Oranı",
            )
            odor_fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(odor_fig, width="stretch")
        with odor_right:
            st.dataframe(odor_risk_df, width="stretch", hide_index=True)

    st.info(
        "Bu sistem eğitim amaçlıdır. Gerçek hayatta uzman doğrulaması olmadan mantar tüketmeyin."
    )
