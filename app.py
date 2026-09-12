import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Skincare Recommender", page_icon="🧴", layout="wide")

DATA_DIR = "data/processed"
MODEL_DIR = "models"


# ---------- Loaders (cached so files are only read/parsed once per session) ----------

@st.cache_data
def load_products():
    df = pd.read_parquet(os.path.join(DATA_DIR, "products_app.parquet"))
    return df.reset_index(drop=True)


@st.cache_data
def load_sample_users():
    return pd.read_parquet(os.path.join(DATA_DIR, "sample_users.parquet"))


@st.cache_resource
def load_svd_model():
    with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_tfidf_vectorizer():
    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_tfidf_matrix(_vectorizer, products_df):
    # Recompute the TF-IDF matrix from the saved vectorizer + product text.
    # This avoids having to ship a huge NxN cosine_sim_matrix.npy file.
    text_features = products_df["ingredients_text"] + " " + products_df["highlights_text"]
    return _vectorizer.transform(text_features)


# ---------- Recommendation logic ----------

def recommend_content_based(product_name, products_df, tfidf_matrix, top_n=5):
    idx_matches = products_df[products_df["product_name"] == product_name].index
    if len(idx_matches) == 0:
        return None
    idx = idx_matches[0]

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    ranked = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    ranked = [r for r in ranked if r[0] != idx][:top_n]

    result = products_df.iloc[[i for i, _ in ranked]][
        ["product_name", "brand_name", "secondary_category"]
    ].copy()
    result["similarity"] = [round(s, 3) for _, s in ranked]
    return result.reset_index(drop=True)


def recommend_collaborative(author_id, algo, products_df, top_n=5):
    scored = []
    for pid in products_df["product_id"]:
        pred = algo.predict(author_id, pid)
        scored.append((pid, pred.est))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_n]

    top_df = pd.DataFrame(top, columns=["product_id", "predicted_rating"])
    result = top_df.merge(
        products_df[["product_id", "product_name", "brand_name", "secondary_category"]],
        on="product_id",
        how="left",
    )
    result["predicted_rating"] = result["predicted_rating"].round(2)
    return result[["product_name", "brand_name", "secondary_category", "predicted_rating"]]


# ---------- UI ----------

st.title("🧴 Sistem Rekomendasi Produk Skincare")
st.caption(
    "Final Project — Machine Learning | Content-Based Filtering (TF-IDF) "
    "& Collaborative Filtering (SVD)"
)

products = load_products()

tab1, tab2 = st.tabs(
    ["🔍 Rekomendasi Berdasarkan Produk", "👤 Rekomendasi Personalized (User)"]
)

with tab1:
    st.subheader("Cari produk yang mirip berdasarkan kandungan bahan & highlight")
    tfidf_vectorizer = load_tfidf_vectorizer()
    tfidf_matrix = build_tfidf_matrix(tfidf_vectorizer, products)

    product_choice = st.selectbox(
        "Pilih produk favoritmu:", products["product_name"].sort_values().unique()
    )
    top_n_cb = st.slider("Jumlah rekomendasi", 3, 10, 5, key="cb_slider")

    if st.button("Cari Produk Mirip"):
        result = recommend_content_based(product_choice, products, tfidf_matrix, top_n=top_n_cb)
        if result is None:
            st.warning("Produk tidak ditemukan.")
        else:
            st.dataframe(result, use_container_width=True)

with tab2:
    st.subheader("Rekomendasi personal berdasarkan riwayat rating pengguna")
    st.caption(
        "Demo memakai sampel user_id dari data training — model SVD hanya bisa "
        "memprediksi untuk user yang sudah ada di data training (bukan user baru)."
    )

    svd_model = load_svd_model()
    sample_users = load_sample_users()

    user_choice = st.selectbox("Pilih contoh User ID:", sample_users["author_id"].tolist())
    top_n_cf = st.slider("Jumlah rekomendasi", 3, 10, 5, key="cf_slider")

    if st.button("Buat Rekomendasi"):
        with st.spinner("Menghitung rekomendasi..."):
            result = recommend_collaborative(user_choice, svd_model, products, top_n=top_n_cf)
        st.dataframe(result, use_container_width=True)

st.divider()
st.caption(
    "Model: SVD (Collaborative Filtering) + TF-IDF Cosine Similarity (Content-Based) | "
    "Dataset: Sephora Products and Skincare Reviews (Kaggle)"
)
