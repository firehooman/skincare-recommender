import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Skincare Recommender", page_icon="૮₍␥ • ⩊ • ␥₎ა", layout="wide")

SPARKLE_LINE = "⋆˚☆˖°⋆｡° ✮˖ ࣪ ⊹⋆.˚"
HEART_LINE = "˖⁺‧₊˚♡˚₊‧⁺˖"
FLOWER = "⋆˚✿˖°"
CAT_FACE = "૮₍ •⩊ •₎ა"

QUIZ_QUESTIONS = [
    {
        "prompt": "Beberapa jam setelah cuci muka, kulitmu terasa...",
        "options": [
            ("Kering dan sedikit ketarik", "dry"),
            ("Berminyak, terutama di dahi & hidung", "oily"),
            ("Berminyak di zona T, tapi kering di pipi", "combination"),
            ("Nyaman, tidak kering atau berminyak", "normal"),
        ],
    },
    {
        "prompt": "Seberapa sering kamu berjerawat atau berkomedo?",
        "options": [
            ("Sering, hampir di seluruh wajah", "oily"),
            ("Jarang sekali", "dry"),
            ("Kadang muncul, tapi cuma di area tertentu", "combination"),
            ("Sesekali, tidak terlalu mengganggu", "normal"),
        ],
    },
    {
        "prompt": "Bagaimana ukuran pori-pori di wajahmu?",
        "options": [
            ("Besar dan terlihat jelas, terutama di hidung", "oily"),
            ("Kecil, hampir tidak terlihat", "dry"),
            ("Bervariasi antar area wajah", "combination"),
            ("Halus dan cenderung merata", "normal"),
        ],
    },
    {
        "prompt": "Setelah seharian tanpa pakai skincare apapun, wajahmu jadi...",
        "options": [
            ("Kaku, kadang sedikit mengelupas", "dry"),
            ("Mengkilap di seluruh wajah", "oily"),
            ("Mengkilap di dahi/hidung, biasa saja di pipi", "combination"),
            ("Baik-baik saja, tidak ada masalah berarti", "normal"),
        ],
    },
]

SKIN_TYPE_LABELS = {
    "dry": "Kulit Kering",
    "oily": "Kulit Berminyak",
    "combination": "Kulit Kombinasi",
    "normal": "Kulit Normal",
}

SIDE_ART = r"""⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣤⣤⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⠋⠀⠀⠙⢿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣸⡇⠀⠀⠀⠀⠀⠙⢿⣦⡀⠀⠀⢀⣀⣀⣠⣤⣀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣿⠇⠀⠀⠀⠀⠀⠀⠀⠙⠿⠿⠟⠛⠛⠋⠉⠉⠛⣷⡄
⠀⠀⠀⠀⠀⠀⠀⢠⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇
⠀⠀⠀⠀⣀⣤⣶⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡿⠃
⠀⣠⣶⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡿⠃⠀
⢸⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡿⠁⠀⠀
⢸⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⡀⠀⠀
⠀⠙⠿⣶⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⡄⠀
⠀⠀⠀⠀⠉⠛⠿⣶⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⡄
⠀⠀⠀⠀⠀⠀⠀⠘⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇
⠀⠀⠀⠀⠀⠀⠀⠀⣿⡆⠀⠀⠀⠀⠀⠀⠀⣠⣶⣶⣦⣤⣤⣄⣀⣀⣤⡿⠃
⠀⠀⠀⠀⠀⠀⠀⠀⢹⡇⠀⠀⠀⠀⠀⣠⣾⠏⠀⠀⠀⠈⠉⠉⠙⠛⠉⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣄⠀⠀⣠⣾⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠛⠛⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀"""

FOOTER_ART = r"""⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣎⠱⣲⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠤⠒⠒⠒⠒⠤⢄⣈⠈⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢀⡤⠒⠝⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠲⢄⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⡴⠋⠀⠀⠀⠀⣀⠀⠀⠀⠀⠀⠀⢠⣢⠐⡄⠀⠉⠑⠒⠒⠒⣄
⠀⠀⠀⣀⠴⠋⠀⠀⠀⡎⢀⣘⠿⠀⠀⢠⣀⢄⡦⠀⣛⣐⢸⠀⠀⠀⠀⠀⠀⢘
⡠⠒⠉⠀⠀⠀⠀⠀⡰⢅⠣⠤⠘⠀⠀⠀⠀⠀⠀⢀⣀⣤⡋⠙⠢⢄⣀⣀⡠⠊
⢇⠀⠀⠀⠀⠀⢀⠜⠁⠀⠉⡕⠒⠒⠒⠒⠒⠛⠉⠹⡄⣀⠘⡄⠀⠀⠀⠀⠀⠀
⠀⠑⠂⠤⠔⠒⠁⠀⠀⡎⠱⡃⠀⠀⡄⠀⠄⠀⠀⠠⠟⠉⡷⠁⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⠤⠤⠴⣄⡸⠤⣄⠴⠤⠴⠄⠼⠀⠀⠀⠀⠀⠀⠀⠀"""

DATA_DIR = "data/processed"
MODEL_DIR = "models"

# ---------- Light custom styling on top of the theme in .streamlit/config.toml ----------
st.markdown(
    """
    <style>
    .hero {
        padding: 1.75rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #F7DCD6 0%, #FBEFEC 100%);
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin-bottom: 0.25rem; }
    .hero p { color: #6B5555; font-size: 1.05rem; margin: 0; }
    .sparkle-top, .sparkle-bottom {
        text-align: center;
        color: #C97B84;
        letter-spacing: 0.15em;
        font-size: 0.95rem;
        margin: 0;
    }
    .sparkle-top { margin-bottom: 0.3rem; }
    .sparkle-bottom { margin-top: 0.3rem; }
    .section-flourish {
        color: #C97B84;
        font-size: 0.85rem;
        letter-spacing: 0.1em;
        margin-bottom: 0.2rem;
    }
    .cat-face {
        text-align: center;
        font-size: 3.5rem;
        color: #C97B84;
        margin: 0.5rem 0 0.2rem 0;
    }
    .quiz-result {
        text-align: center;
        border: 1px solid #EBDAD5;
        border-radius: 14px;
        padding: 1.2rem;
        background-color: #FBEFEC;
        margin-top: 1rem;
    }
    .quiz-result h3 { color: #C97B84; margin-bottom: 0.2rem; }
    .ascii-art {
        white-space: pre;
        font-family: monospace;
        line-height: 1;
        font-size: 0.45rem;
        color: #C97B84;
        text-align: center;
        overflow-x: auto;
        margin: 0.5rem 0;
    }
    .product-card {
        border: 1px solid #EBDAD5;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        background-color: #FFFFFF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .product-card h4 { margin: 0 0 0.15rem 0; }
    .product-card .brand { color: #9A8888; font-size: 0.85rem; margin-bottom: 0.4rem; }
    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background-color: #FBEFEC;
        color: #C97B84;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Loaders (cached so files are only read/parsed once per session) ----------

@st.cache_data
def load_products():
    df = pd.read_parquet(os.path.join(DATA_DIR, "products_app.parquet"))
    return df.reset_index(drop=True)


@st.cache_data
def load_sample_users():
    return pd.read_parquet(os.path.join(DATA_DIR, "sample_users.parquet"))


@st.cache_data
def load_skin_stats():
    return pd.read_parquet(os.path.join(DATA_DIR, "product_skin_type_stats.parquet"))


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
    text_features = products_df["ingredients_text"] + " " + products_df["highlights_text"]
    return _vectorizer.transform(text_features)


# ---------- Recommendation logic (cached by input so repeat clicks are instant) ----------

@st.cache_data
def recommend_content_based(product_name, products_df, _tfidf_matrix, top_n=5):
    idx_matches = products_df[products_df["product_name"] == product_name].index
    if len(idx_matches) == 0:
        return None
    idx = idx_matches[0]

    sim_scores = cosine_similarity(_tfidf_matrix[idx], _tfidf_matrix).flatten()
    ranked = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    ranked = [r for r in ranked if r[0] != idx][:top_n]

    result = products_df.iloc[[i for i, _ in ranked]][
        ["product_name", "brand_name", "secondary_category"]
    ].copy()
    result["score"] = [round(s, 3) for _, s in ranked]
    result["score_label"] = "Similarity"
    return result.reset_index(drop=True)


@st.cache_data
def recommend_collaborative(author_id, _algo, products_df, top_n=5):
    scored = []
    for pid in products_df["product_id"]:
        pred = _algo.predict(author_id, pid)
        scored.append((pid, pred.est))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_n]

    top_df = pd.DataFrame(top, columns=["product_id", "score"])
    result = top_df.merge(
        products_df[["product_id", "product_name", "brand_name", "secondary_category"]],
        on="product_id",
        how="left",
    )
    result["score"] = result["score"].round(2)
    result["score_label"] = "Predicted rating"
    return result[["product_name", "brand_name", "secondary_category", "score", "score_label"]]


@st.cache_data
def recommend_by_skin_type(skin_type, category, products_df, skin_stats_df, top_n=5):
    filtered = skin_stats_df[skin_stats_df["skin_type"] == skin_type]

    result = filtered.merge(
        products_df[["product_id", "product_name", "brand_name", "secondary_category"]],
        on="product_id",
        how="left",
    )

    if category != "Semua Kategori":
        result = result[result["secondary_category"] == category]

    result = result.sort_values(["avg_rating", "review_count"], ascending=False).head(top_n)
    result["score"] = result["avg_rating"].round(2)
    result["score_label"] = f"Avg rating ({skin_type})"
    return result[["product_name", "brand_name", "secondary_category", "score", "score_label"]]


def render_product_cards(df, columns=2):
    if df is None or df.empty:
        st.warning("Tidak ada rekomendasi ditemukan.")
        return

    cols = st.columns(columns)
    for i, row in df.reset_index(drop=True).iterrows():
        with cols[i % columns]:
            st.markdown(
                f"""
                <div class="product-card">
                    <h4>{row['product_name']}</h4>
                    <div class="brand">{row['brand_name']}</div>
                    <span class="badge">{row['secondary_category']}</span>
                    <span class="badge">★ {row['score']} {row['score_label']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------- Sidebar: project info + model performance ----------

with st.sidebar:
    st.markdown(f'<div class="section-flourish">{FLOWER}</div>', unsafe_allow_html=True)
    st.header("Tentang Model")
    st.caption("Ringkasan performa model dari tahap evaluasi.")
    c1, c2 = st.columns(2)
    c1.metric("RMSE (test)", "0.95")
    c2.metric("Precision@10", "0.72")
    c1.metric("Overfit gap", "0.18", help="Selisih Train RMSE vs Test RMSE — makin kecil makin baik")
    c2.metric("Model", "SVD")
    st.markdown(f'<p style="text-align:center;color:#C97B84;">{HEART_LINE}</p>', unsafe_allow_html=True)
    st.markdown(
        "**Dataset:** Sephora Products & Skincare Reviews (Kaggle)  \n"
        "**Metode:** Content-Based (TF-IDF) & Collaborative Filtering (SVD)"
    )
    st.markdown(f'<div class="ascii-art">{SIDE_ART}</div>', unsafe_allow_html=True)

# ---------- Hero header ----------

st.markdown(
    f"""
    <p class="sparkle-top">{SPARKLE_LINE}</p>
    <div class="hero">
        <h1 style="text-align:center;">Skincare Recommender</h1>
        <p style="text-align:center;">Temukan produk skincare yang cocok — berdasarkan kandungan bahan, atau berdasarkan pola rating pengguna lain.</p>
    </div>
    <p class="sparkle-bottom">{HEART_LINE}</p>
    """,
    unsafe_allow_html=True,
)

products = load_products()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Kenali Jenis Kulitmu", "Berdasarkan Jenis Kulit", "Personalized (User)", "Berdasarkan Produk"]
)

with tab1:
    st.markdown(f'<div class="cat-face">{CAT_FACE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#6B5555;">Belum tahu jenis kulitmu? '
        "Jawab pertanyaan singkat ini dulu, yuk!</p>",
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="sparkle-bottom">{HEART_LINE}</p>', unsafe_allow_html=True)

    answers = []
    for i, q in enumerate(QUIZ_QUESTIONS):
        st.markdown(f'<div class="section-flourish">{FLOWER}</div>', unsafe_allow_html=True)
        choice = st.radio(
            q["prompt"],
            options=q["options"],
            format_func=lambda opt: opt[0],
            key=f"quiz_q{i}",
        )
        answers.append(choice[1])

    if st.button("Cek Jenis Kulitku", use_container_width=True):
        tally = pd.Series(answers).value_counts()
        result_type = tally.idxmax()
        result_label = SKIN_TYPE_LABELS.get(result_type, result_type)

        st.markdown(
            f"""
            <div class="quiz-result">
                <div class="cat-face" style="font-size:2.2rem;">{CAT_FACE}</div>
                <h3>{result_label}</h3>
                <p style="color:#6B5555;">Coba cek tab <b>"Berdasarkan Jenis Kulit"</b> dan pilih
                <b>{result_type}</b> untuk lihat produk yang paling cocok!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab2:
    st.markdown(f'<div class="section-flourish">{FLOWER}</div>', unsafe_allow_html=True)
    st.subheader("Cocok untuk pemula: pilih jenis kulitmu")
    st.caption(
        "Menampilkan produk dengan rating tertinggi dari pengguna lain yang punya "
        "jenis kulit sama denganmu."
    )

    skin_stats = load_skin_stats()
    skin_type_options = sorted(skin_stats["skin_type"].dropna().unique().tolist())
    category_options = ["Semua Kategori"] + sorted(products["secondary_category"].dropna().unique().tolist())

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        skin_type_choice = st.selectbox("Jenis kulitmu:", skin_type_options)
    with col_b:
        category_choice = st.selectbox("Kategori produk (opsional):", category_options)
    with col_c:
        top_n_skin = st.slider("Jumlah", 3, 10, 5, key="skin_slider")

    if st.button("Cari Produk untuk Jenis Kulitku", use_container_width=True):
        result = recommend_by_skin_type(
            skin_type_choice, category_choice, products, skin_stats, top_n=top_n_skin
        )
        render_product_cards(result)

with tab3:
    st.markdown(f'<div class="section-flourish">{FLOWER}</div>', unsafe_allow_html=True)
    st.subheader("Rekomendasi personal berdasarkan riwayat rating pengguna")
    st.caption(
        "Demo memakai sampel pengguna dari data training — model SVD hanya bisa "
        "memprediksi untuk pengguna yang sudah ada di data training (bukan pengguna baru)."
    )

    svd_model = load_svd_model()
    sample_users = load_sample_users()

    col_a, col_b = st.columns([3, 1])
    with col_a:
        user_choice = st.selectbox("Pilih contoh User ID:", sample_users["author_id"].tolist())
    with col_b:
        top_n_cf = st.slider("Jumlah", 3, 10, 5, key="cf_slider")

    selected_user = sample_users[sample_users["author_id"] == user_choice].iloc[0]
    user_skin_label = SKIN_TYPE_LABELS.get(selected_user["skin_type"], "Tidak diketahui")
    st.caption(f"Kondisi kulit pengguna ini: **{user_skin_label}**")

    if st.button("Buat Rekomendasi", use_container_width=True):
        with st.spinner("Menghitung rekomendasi..."):
            result = recommend_collaborative(user_choice, svd_model, products, top_n=top_n_cf)
        render_product_cards(result)

with tab4:
    st.markdown(f'<div class="section-flourish">{FLOWER}</div>', unsafe_allow_html=True)
    st.subheader("Cari produk yang mirip berdasarkan kandungan bahan & highlight")
    tfidf_vectorizer = load_tfidf_vectorizer()
    tfidf_matrix = build_tfidf_matrix(tfidf_vectorizer, products)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        product_choice = st.selectbox(
            "Pilih produk favoritmu:", products["product_name"].sort_values().unique()
        )
    with col_b:
        top_n_cb = st.slider("Jumlah", 3, 10, 5, key="cb_slider")

    if st.button("Cari Produk Mirip", use_container_width=True):
        result = recommend_content_based(product_choice, products, tfidf_matrix, top_n=top_n_cb)
        render_product_cards(result)

st.markdown(
    f"""
    <div class="ascii-art">{FOOTER_ART}</div>
    <p style="text-align:center;color:#9A8888;font-size:0.85rem;">
        Final Project Machine Learning — SVD (Collaborative Filtering) +
        TF-IDF Cosine Similarity (Content-Based)
    </p>
    """,
    unsafe_allow_html=True,
)
