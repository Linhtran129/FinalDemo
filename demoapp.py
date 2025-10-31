import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os
from scipy.sparse import csr_matrix

# ==============================
# 🧠 1️⃣ Load model
# ==============================
@st.cache_resource
def load_model():
    model_data = pickle.load(open("knn_model_tuned.pkl", "rb"))
    model_data["i_id2idx"] = {str(k): v for k, v in model_data["i_id2idx"].items()}
    model_data["i_idx2id"] = {v: str(k) for k, v in model_data["i_id2idx"].items()}
    return model_data

model_data = load_model()
knn_model = model_data["model"]
i_id2idx = model_data["i_id2idx"]
i_idx2id = model_data["i_idx2id"]

# ==============================
# 📂 2️⃣ Load dữ liệu
# ==============================
@st.cache_data(show_spinner=True)
def load_data_from_drive(sample_size=5000):
    file_id = "1tbu4qD5Pnmgs7JypumLZLW6MKuqYUH4d"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    local_path = "events.json"

    if not os.path.exists(local_path):
        with st.spinner("🔽 Đang tải dữ liệu..."):
            gdown.download(url, local_path, quiet=False)

    with open(local_path, "r", encoding="utf-8") as f:
        start = f.read(200).strip()

    dfs = []
    try:
        if start.startswith("["):
            df = pd.read_json(local_path)
        else:
            for chunk in pd.read_json(local_path, lines=True, chunksize=200000):
                dfs.append(chunk)
            df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"❌ Lỗi đọc file JSON: {e}")
        st.stop()

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    return df

# ==============================
# ⚙️ 3️⃣ Tiền xử lý
# ==============================
df = load_data_from_drive()
event_weight = {"view": 1, "addtocart": 3, "transaction": 5}
df["weight"] = df["event"].map(event_weight)
df["itemid"] = df["itemid"].astype(str)
df = df[df["itemid"].isin(i_id2idx.keys())]

# Tạo giá ngẫu nhiên (vì dataset gốc không có)
np.random.seed(42)
price_map = {iid: np.random.randint(50, 500) * 1000 for iid in df["itemid"].unique()}
df["price"] = df["itemid"].map(price_map)

# ==============================
# 💡 4️⃣ Gợi ý sản phẩm
# ==============================
def get_top_n_knn(seed_item, top_n=10):
    if seed_item not in i_id2idx:
        return []
    item_idx = i_id2idx[seed_item]
    fit_X = knn_model._fit_X
    item_vector = fit_X[item_idx].reshape(1, -1)
    distances, indices = knn_model.kneighbors(item_vector, n_neighbors=top_n + 1)
    indices = indices.flatten()[1:]
    return [i_idx2id[idx] for idx in indices if idx in i_idx2id]

# ==============================
# 🎨 5️⃣ Giao diện
# ==============================
st.set_page_config(page_title="PetShop Recommender", page_icon="🐾", layout="wide")
st.title("🐾 PetShop Product Recommender System")

menu = st.sidebar.radio("📋 Chọn trang:", ["Trang chủ", "Giỏ hàng", "Thanh toán"])

if "cart" not in st.session_state:
    st.session_state.cart = []

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

# ==============================
# 🏠 Trang chủ
# ==============================
if menu == "Trang chủ":
    visitor_ids = sorted(df["visitorid"].unique())
    prev_visitor = st.session_state.get("current_visitor")
    visitor_id = st.selectbox("👤 Chọn Visitor ID:", visitor_ids)

    # reset gợi ý khi đổi visitor
    if prev_visitor != visitor_id:
        st.session_state.recommendations = []
        st.session_state["current_visitor"] = visitor_id

    user_items = df[df["visitorid"] == visitor_id]["itemid"].unique()
    st.write(f"Người dùng **{visitor_id}** đã tương tác với {len(user_items)} sản phẩm.")

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
            st.write(f"💰 Giá: **{item_info['price']:,} VND**")

        # Nút gợi ý
        if st.button("🎯 Gợi ý sản phẩm tương tự", use_container_width=True):
            rec_items = get_top_n_knn(seed_item, 10)
            rec_df = df[df["itemid"].isin(rec_items)].drop_duplicates("itemid")[
                ["itemid", "item_name", "item_description", "image_url", "price"]
            ]
            st.session_state.recommendations = rec_df.to_dict("records")

        # Hiển thị gợi ý
        if st.session_state.recommendations:
            st.subheader("✨ Gợi ý cho bạn")
            rec_df = pd.DataFrame(st.session_state.recommendations)

            for row_start in range(0, len(rec_df), 4):
                cols = st.columns(4)
                for i, (_, row) in enumerate(rec_df.iloc[row_start:row_start + 4].iterrows()):
                    with cols[i]:
                        st.image(row["image_url"], width=180)
                        st.markdown(f"**{row['item_name']}**")
                        st.caption(row["item_description"][:100] + "...")
                        st.write(f"💰 **{row['price']:,} VND**")
                        if st.button("🛍️ Thêm vào giỏ", key=f"rec_{row['itemid']}"):
                            found = False
                            for item in st.session_state.cart:
                                if item["itemid"] == row["itemid"]:
                                    item["quantity"] += 1
                                    found = True
                                    break
                            if not found:
                                st.session_state.cart.append({
                                    "itemid": row["itemid"],
                                    "item_name": row["item_name"],
                                    "image_url": row["image_url"],
                                    "quantity": 1,
                                    "price": row["price"]
                                })
                            st.success(f"✅ Đã thêm {row['item_name']} vào giỏ hàng")

# ==============================
# 🛒 Trang Giỏ hàng
# ==============================
elif menu == "Giỏ hàng":
    st.header("🛒 Giỏ hàng của bạn")

    if not st.session_state.cart:
        st.info("Giỏ hàng trống. Hãy chọn sản phẩm ở Trang chủ nhé!")
    else:
        total_price = 0
        for idx, item in enumerate(st.session_state.cart):
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    st.image(item["image_url"], width=100)
                with col2:
                    st.markdown(f"**{item['item_name']}**")
                    st.write(f"💰 {item['price']:,} VND")
                    qty_col1, qty_col2, qty_col3 = st.columns(3)
                    with qty_col1:
                        if st.button("-", key=f"minus_{idx}") and item["quantity"] > 1:
                            item["quantity"] -= 1
                    with qty_col2:
                        st.write(f"**{item['quantity']}**")
                    with qty_col3:
                        if st.button("+", key=f"plus_{idx}"):
                            item["quantity"] += 1
                with col3:
                    if st.button("🗑️ Xóa", key=f"del_{idx}"):
                        st.session_state.cart.pop(idx)
                        st.experimental_rerun()
            st.divider()
            total_price += item["quantity"] * item["price"]

        st.markdown(f"### 💰 Tổng cộng: **{total_price:,} VND**")
        st.divider()

        if st.button("💳 Tiến hành thanh toán", use_container_width=True):
            st.session_state.total_price = total_price
            st.session_state.menu = "Thanh toán"
            st.experimental_set_query_params(menu="Thanh toán")
            st.experimental_rerun()

# ==============================
# 💰 Trang Thanh toán
# ==============================
elif menu == "Thanh toán":
    st.header("💳 Thanh toán")

    if not st.session_state.cart:
        st.warning("Giỏ hàng trống. Vui lòng quay lại Trang chủ để chọn sản phẩm.")
