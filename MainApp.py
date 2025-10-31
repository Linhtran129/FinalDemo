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
    """Load KNN model + mapping itemid <-> index"""
    model_data = pickle.load(open("knn_model_tuned.pkl", "rb"))
    # đảm bảo key là string để khớp với dữ liệu
    model_data["i_id2idx"] = {str(k): v for k, v in model_data["i_id2idx"].items()}
    model_data["i_idx2id"] = {v: str(k) for k, v in model_data["i_id2idx"].items()}
    return model_data

model_data = load_model()
knn_model = model_data["model"]
i_id2idx = model_data["i_id2idx"]
i_idx2id = model_data["i_idx2id"]


# ==============================
# 📂 2️⃣ Load dữ liệu từ Google Drive
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
# ⚙️ 3️⃣ Tiền xử lý dữ liệu
# ==============================
df = load_data_from_drive()
event_weight = {"view": 1, "addtocart": 3, "transaction": 5}
df["weight"] = df["event"].map(event_weight)
df["itemid"] = df["itemid"].astype(str)
df = df[df["itemid"].isin(i_id2idx.keys())]  # chỉ giữ item có trong model


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
# 🎨 5️⃣ Giao diện Streamlit
# ==============================
st.set_page_config(page_title="PetShop Recommender", page_icon="🐾", layout="wide")
st.title("🐾 PetShop Product Recommender System")

# --- Sidebar: Giỏ hàng + chuyển trang ---
menu = st.sidebar.radio("📋 Chọn trang:", ["Trang chủ", "Giỏ hàng", "Thanh toán"])

if "cart" not in st.session_state:
    st.session_state.cart = []

# ==============================
# 🏠 Trang chủ
# ==============================
# ==============================
# 🏠 Trang chủ
# ==============================
if menu == "Trang chủ":
    visitor_ids = sorted(df["visitorid"].unique())
    visitor_id = st.selectbox("👤 Chọn Visitor ID:", visitor_ids)

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

        # Nút gợi ý — chỉ cần bấm 1 lần, lưu vào session
        if st.button("🎯 Gợi ý sản phẩm tương tự", use_container_width=True):
            rec_items = get_top_n_knn(seed_item, 10)
            rec_df = df[df["itemid"].isin(rec_items)].drop_duplicates("itemid")[["itemid", "item_name", "item_description", "image_url"]]
            st.session_state["recommendations"] = rec_df.to_dict("records")

        # Hiển thị lại gợi ý từ session nếu có
        if "recommendations" in st.session_state and st.session_state["recommendations"]:
            st.subheader("✨ Gợi ý cho bạn")
            rec_df = pd.DataFrame(st.session_state["recommendations"])

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
                            st.success(f"✅ Đã thêm {row['item_name']} vào giỏ hàng")


# ==============================
# 🛒 Trang Giỏ hàng
# ==============================
elif menu == "Giỏ hàng":
    st.header("🛒 Giỏ hàng của bạn")

    if len(st.session_state.cart) == 0:
        st.info("Giỏ hàng trống. Hãy chọn sản phẩm ở Trang chủ nhé!")
    else:
        total = len(st.session_state.cart)
        st.write(f"Bạn có **{total}** sản phẩm trong giỏ.")
        for item in st.session_state.cart:
            with st.container():
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(item["image_url"], width=100)
                with col2:
                    st.markdown(f"**{item['item_name']}**")
        st.divider()
        if st.button("💳 Tiến hành thanh toán", use_container_width=True):
            st.session_state.page = "Thanh toán"
            st.switch_page("Thanh toán")

# ==============================
# 💰 Trang Thanh toán
# ==============================
elif menu == "Thanh toán":
    st.header("💳 Thanh toán")
    if len(st.session_state.cart) == 0:
        st.warning("Giỏ hàng trống. Vui lòng quay lại Trang chủ để chọn sản phẩm.")
    else:
        name = st.text_input("👤 Họ và tên")
        acc = st.text_input("🏦 Số tài khoản")
        if st.button("✅ Xác nhận thanh toán", use_container_width=True):
            st.success(f"🎉 Thanh toán thành công! Cảm ơn {name} đã mua hàng.")
            st.balloons()
            st.session_state.cart = []
