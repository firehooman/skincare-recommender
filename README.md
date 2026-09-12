# Skincare Recommender System

Final Project Machine Learning — sistem rekomendasi produk skincare yang membantu pengguna (terutama pemula) menemukan produk yang cocok, baik berdasarkan kemiripan kandungan produk, riwayat rating personal, maupun jenis kulit.

Dataset: [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) (Kaggle)

## Fitur

1. **Berdasarkan Produk** — content-based filtering (TF-IDF + cosine similarity) dari kandungan bahan & highlight produk
2. **Personalized (User)** — collaborative filtering (SVD) berdasarkan riwayat rating pengguna
3. **Berdasarkan Jenis Kulit** — rekomendasi produk dengan rating tertinggi dari pengguna lain yang punya jenis kulit sama
4. **Kenali Jenis Kulitmu** — mini-kuis untuk membantu pengguna yang belum tahu jenis kulitnya

## Metodologi Singkat

- **ETL**: ekstraksi & pembersihan data produk dan review Sephora
- **EDA**: analisis distribusi rating, kategori produk, dan pengaruh jenis kulit terhadap rating
- **Modeling**: perbandingan Popularity Baseline, Content-Based Filtering (TF-IDF), dan Collaborative Filtering (SVD), dengan Hyperparameter Tuning (GridSearchCV) untuk mengatasi overfitting
- **Evaluation**: RMSE, MAE, Precision@K, Recall@K, serta analisis overfitting/underfitting dan performa per kategori produk

| Model | RMSE (test) | Catatan |
|---|---|---|
| SVD default | 0.93 | Baseline sebelum tuning |
| **SVD tuned** | **0.95** | Gap overfitting turun drastis (0.72 → 0.18) |

## Live App

**[Buka aplikasi di sini](https://skincare-recommender-xp7q6fxo5kndrqfwftejcz.streamlit.app/#kulit-berminyak)**

## Cara Menjalankan di Lokal

1. Clone repository ini:
   ```bash
   git clone https://github.com/firehooman/skincare-recommender.git
   cd skincare-recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

4. Buka browser ke `http://localhost:8501`

## Struktur Folder

```
skincare-recommender/
├── .streamlit/
│   └── config.toml              # tema tampilan aplikasi
├── app.py                       # source code aplikasi Streamlit
├── requirements.txt             # dependencies Python
├── packages.txt                 # dependencies sistem (untuk build scikit-surprise)
├── final_project.ipynb          # notebook lengkap (ETL, EDA, Modeling, Evaluation)
├── data/
│   └── processed/
│       ├── products_app.parquet            # metadata produk untuk aplikasi
│       ├── sample_users.parquet            # contoh user_id untuk demo personalized
│       └── product_skin_type_stats.parquet # rata-rata rating produk per jenis kulit
└── models/
    ├── svd_model.pkl             # model SVD hasil tuning
    └── tfidf_vectorizer.pkl      # vectorizer TF-IDF untuk content-based filtering
```

## Ringkasan Hasil

- RMSE test set: **0.95** (rata-rata prediksi meleset ±0.95 poin dari rating asli, skala 1–5)
- Precision@10: **0.72** (72% dari 10 rekomendasi teratas relevan bagi pengguna)
- Tidak ada indikasi overfitting setelah tuning (gap Train-Test RMSE: 0.18)

## Author

Zinniarethie Andari Kostiene
Final Project Machine Learning ⋆˚꩜｡