import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown
from scipy.sparse import csr_matrix

# ==============================
# ⚙️ 1️⃣ Load data an toàn từ Google Drive
# ==============================

@st.cache_data(show_spinner=True)
def load_data(sample_size=5000):
    # 🧩 Link file Google Drive
    file_id = "1tbu4qD5Pnmgs7JypumLZLW6MKuqYUH4d"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    local_path = "events.json"

    # Nếu file chưa tồn tại thì tải về
    if not os.path.exists(local_path):
        with st.spinner("🔽 Đang tải dữ liệu từ Google Drive..."):
            gdown.download(url, local_path, quiet=False)

    # Đọc JSON theo dòng để tránh tràn RAM
    dfs = []
    for chunk in pd.read_json(local_path, lines=True, chunksize=200000):
        dfs.append(chunk[["visitorid", "event", "itemid", "item_name", "item_description", "image_url"]])
    df = pd.concat(dfs, ignore_index=True)

    # Lấy mẫu nhẹ để chạy nhanh (có thể tăng nếu deploy mạnh)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df


df = load_data()

# ==============================
# 🧮 2️⃣ Xử lý dữ liệu & ma trận user-item
# ==============================

event_weight = {"view": 1, "addtocart": 3, "transaction": 5}
df["weight"] = df["event"].map(event_weight)

user_item_matrix = df.pivot_table(index="visitorid", columns="itemid", values="weight", fill_value=0)
user_item_csr = csr_matrix(user_item_matrix.values)

i_id2idx = {id_: idx for idx, id_ in enumerate(user_item_matrix.columns)}
i_idx2id = {idx: id_ for id_, idx in i_id2idx.items()}

# ==============================
# 🧠 3️⃣ Load model
# ==============================

@st.cache_resource
def load_model():
    return pickle.load(open("knn_model_tuned.pkl", "rb"))

knn_model = load_model()

def get_top_n_knn(seed_item, top_n=10):
    """Top-N sản phẩm tương tự (Collaborative Filtering - KNN)."""
    item_idx = i_id2idx.get(seed_item)
    if item_idx is None:
        return []

    item_vector = user_item_csr[:, item_idx].T
    distances, indices = knn_model.kneighbors(item_vector, n_neighbors=top_n + 1)
    indices = indices.flatten()[1:top_n + 1]
    similar_items = [i_idx2id[i] for i in indices]
    return similar_items

# ==============================
# 🛍️ 4️⃣ Giao diện Streamlit
# ==============================

st.set_page_config(page_title="PetShop Recommender", page_icon="🐾", layout="wide")
st.title("🐾 PetShop Product Recommender System")

# --- Giỏ hàng ---
if "cart" not in st.session_state:
    st.session_state.cart = []

st.sidebar.header("🛒 Giỏ hàng của bạn")
if len(st.session_state.cart) == 0:
    st.sidebar.info("Giỏ hàng trống.")
else:
    for item in st.session_state.cart:
        st.sidebar.image(item["image_url"], width=80)
        st.sidebar.write(f"**{item['item_name']}**")
        st.sidebar.caption(f"Số lượng: {item['quantity']}")
    if st.sidebar.button("💳 Thanh toán"):
        st.sidebar.success("✅ Thanh toán thành công!")
        st.session_state.cart = []

# --- Chọn người dùng ---
visitor_ids = sorted(df["visitorid"].unique())
visitor_id = st.selectbox("👤 Chọn Visitor ID:", visitor_ids)

user_items = df[df["visitorid"] == visitor_id]["itemid"].unique()
st.write(f"Người dùng **{visitor_id}** đã tương tác với {len(user_items)} sản phẩm.")

# --- Chọn sản phẩm ---
if len(user_items) > 0:
    seed_item = st.selectbox("📦 Chọn sản phẩm để xem gợi ý:", user_items)

    item_info = df[df["itemid"] == seed_item].iloc[0]
    st.subheader("📍 Sản phẩm đã xem")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(item_info["image_url"], width=200)
    with col2:
        st.markdown(f"**{item_info['item_name']}**")
        st.caption(item_info["item_description"][:400])

    top_n = st.slider("🔢 Số lượng sản phẩm muốn gợi ý:", 5, 20, 10)

    if st.button("🎯 Gợi ý sản phẩm tương tự"):
        rec_items = get_top_n_knn(seed_item, top_n)
        rec_df = df[df["itemid"].isin(rec_items)].drop_duplicates("itemid")[["itemid", "item_name", "item_description", "image_url"]]

        st.subheader("✨ Gợi ý cho bạn")
        for row_start in range(0, len(rec_df), 4):
            cols = st.columns(4)
            for i, (_, row) in enumerate(rec_df.iloc[row_start:row_start + 4].iterrows()):
                with cols[i]:
                    st.image(row["image_url"], width=180)
                    st.markdown(f"**{row['item_name']}**")
                    st.caption(row["item_description"][:100] + "...")
                    if st.button("🛍️ Thêm vào giỏ", key=f"rec_{row['itemid']}"):
                        st.session_state.cart.append({
                            "itemid": row["itemid"],
                            "item_name": row["item_name"],
                            "image_url": row["image_url"],
                            "quantity": 1
                        })
                        st.toast(f"✅ Đã thêm {row['item_name']} vào giỏ hàng", icon="🛒")
