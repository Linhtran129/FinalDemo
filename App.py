import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os
from scipy.sparse import csr_matrix

# ==============================
# ⚙️ 1️⃣ Load data an toàn từ Google Drive
# ==============================


# ==============================
# 🧠 1️⃣ Load model (trước khi xử lý dữ liệu)
# ==============================
@st.cache_resource
def load_model():
    """Load KNN model + mapping itemid <-> index"""
    model_data = pickle.load(open("knn_model_tuned.pkl", "rb"))
    # đảm bảo tất cả key là string để khớp với df
    model_data["i_id2idx"] = {str(k): v for k, v in model_data["i_id2idx"].items()}
    model_data["i_idx2id"] = {v: str(k) for k, v in model_data["i_id2idx"].items()}
    return model_data

model_data = load_model()
knn_model = model_data["model"]
i_id2idx = model_data["i_id2idx"]
i_idx2id = model_data["i_idx2id"]



# ==============================
# 📂 2️⃣ Load data từ Google Drive
# ==============================
@st.cache_data(show_spinner=True)
def load_data_from_drive(sample_size=5000):
    file_id = "1tbu4qD5Pnmgs7JypumLZLW6MKuqYUH4d"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    local_path = "events.json"

    if not os.path.exists(local_path):
        with st.spinner("🔽 Đang tải dữ liệu từ Google Drive..."):
            gdown.download(url, local_path, quiet=False)
        st.success("✅ Tải dữ liệu thành công!")

    with open(local_path, "r", encoding="utf-8") as f:
        start = f.read(200).strip()

    dfs = []
    try:
        if start.startswith("["):
            st.info("📄 Phát hiện file dạng JSON array → đọc bình thường (không lines=True)")
            df = pd.read_json(local_path)
        else:
            st.info("📄 Phát hiện file dạng JSON-lines → đọc theo chunks")
            for chunk in pd.read_json(local_path, lines=True, chunksize=200000):
                dfs.append(chunk)
            df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"❌ Lỗi đọc file JSON: {e}")
        st.stop()

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    st.success(f"✅ Đã load {len(df):,} dòng dữ liệu.")
    return df


# ==============================
# ⚙️ 3️⃣ Tiền xử lý dữ liệu
# ==============================
df = load_data_from_drive()
event_weight = {"view": 1, "addtocart": 3, "transaction": 5}
df["weight"] = df["event"].map(event_weight)
df["itemid"] = df["itemid"].astype(str)

# lọc những item có trong model
df = df[df["itemid"].isin(i_id2idx.keys())]


# ==============================
# 💡 4️⃣ Hàm gợi ý sản phẩm
# ==============================
def get_top_n_knn(seed_item, top_n=10):
    """Gợi ý sản phẩm tương tự từ model gốc."""
    if seed_item not in i_id2idx:
        return []

    item_idx = i_id2idx[seed_item]
    fit_X = knn_model._fit_X
    item_vector = fit_X[item_idx].reshape(1, -1)

    distances, indices = knn_model.kneighbors(item_vector, n_neighbors=top_n + 1)
    indices = indices.flatten()[1:]  # bỏ chính nó
    similar_items = [i_idx2id[idx] for idx in indices if idx in i_idx2id]
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
