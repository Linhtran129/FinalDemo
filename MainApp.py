import streamlit as st
import pandas as pd
import pickle
import gdown
import os
import random
from scipy.sparse import csr_matrix

# ==============================
# ⚙️ 1️⃣ Load model
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
# 📂 2️⃣ Load data
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
    if start.startswith("["):
        df = pd.read_json(local_path)
    else:
        for chunk in pd.read_json(local_path, lines=True, chunksize=200000):
            dfs.append(chunk)
        df = pd.concat(dfs, ignore_index=True)

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    # Nếu chưa có cột price thì tạo ngẫu nhiên
    if "price" not in df.columns:
        random.seed(42)
        df["price"] = [random.randint(50000, 500000) for _ in range(len(df))]

    return df


# ==============================
# ⚙️ 3️⃣ Tiền xử lý dữ liệu
# ==============================
df = load_data_from_drive()
event_weight = {"view": 1, "addtocart": 3, "transaction": 5}
df["weight"] = df["event"].map(event_weight)
df["itemid"] = df["itemid"].astype(str)
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

    n_neighbors = min(top_n + 1, fit_X.shape[0])
    distances, indices = knn_model.kneighbors(item_vector, n_neighbors=n_neighbors)
    indices = indices.flatten()[1:]
    similar_items = [i_idx2id[idx] for idx in indices if idx in i_idx2id]

    # Bổ sung random item nếu thiếu
    if len(similar_items) < top_n:
        remaining = list(set(i_id2idx.keys()) - set(similar_items) - {seed_item})
        extra_items = random.sample(remaining, min(top_n - len(similar_items), len(remaining)))
        similar_items.extend(extra_items)
    return similar_items[:top_n]


# ==============================
# 🛍️ 5️⃣ Giao diện chính
# ==============================
st.set_page_config(page_title="PetShop Recommender", page_icon="🐾", layout="wide")

if "cart" not in st.session_state:
    st.session_state.cart = []

if "page" not in st.session_state:
    st.session_state.page = "home"


# ========== 🔹 TRANG 1: HOME ==========
if st.session_state.page == "home":
    st.title("🐾 PetShop Product Recommender System")

    # Sidebar giỏ hàng
    st.sidebar.header("🛒 Giỏ hàng của bạn")
    total = 0
    if len(st.session_state.cart) == 0:
        st.sidebar.info("Giỏ hàng trống.")
    else:
        for item in st.session_state.cart:
            st.sidebar.image(item["image_url"], width=80)
            st.sidebar.write(f"**{item['item_name']}** - {item['price']:,}₫")
            st.sidebar.caption(f"Số lượng: {item['quantity']}")
            total += item["price"] * item["quantity"]
        st.sidebar.markdown(f"### 💰 Tổng cộng: **{total:,}₫**")
        if st.sidebar.button("💳 Thanh toán"):
            st.session_state.page = "checkout"
            st.rerun()

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
            st.write(f"💵 **Giá:** {item_info['price']:,}₫")

        top_n = st.slider("🔢 Số lượng sản phẩm muốn gợi ý:", 5, 20, 10)

        if st.button("🎯 Gợi ý sản phẩm tương tự", key="recommend"):
            rec_items = get_top_n_knn(seed_item, top_n)
            rec_df = df[df["itemid"].isin(rec_items)].drop_duplicates("itemid")[["itemid", "item_name", "item_description", "image_url", "price"]]

            if len(rec_df) == 0:
                st.warning("⚠️ Không tìm thấy sản phẩm tương tự.")
            else:
                st.subheader("✨ Gợi ý cho bạn")
                for row_start in range(0, len(rec_df), 4):
                    cols = st.columns(4)
                    for i, (_, row) in enumerate(rec_df.iloc[row_start:row_start + 4].iterrows()):
                        with cols[i]:
                            st.image(row["image_url"], width=180)
                            st.markdown(f"**{row['item_name']}**")
                            st.caption(row["item_description"][:100] + "...")
                            st.write(f"💵 {row['price']:,}₫")
                            add_key = f"add_{visitor_id}_{row['itemid']}"
                            if st.button("🛍️ Thêm vào giỏ", key=add_key):
                                st.session_state.cart.append({
                                    "itemid": row["itemid"],
                                    "item_name": row["item_name"],
                                    "image_url": row["image_url"],
                                    "price": row["price"],
                                    "quantity": 1
                                })
                                st.toast(f"✅ Đã thêm {row['item_name']} vào giỏ hàng", icon="🛒")
                                st.rerun()


# ========== 🔹 TRANG 2: CHECKOUT ==========
elif st.session_state.page == "checkout":
    st.title("💳 Trang Thanh Toán")

    cart = st.session_state.cart
    if len(cart) == 0:
        st.info("🛒 Giỏ hàng trống, quay lại trang chính để chọn sản phẩm.")
        if st.button("↩️ Quay lại"):
            st.session_state.page = "home"
            st.rerun()
    else:
        st.subheader("🧾 Chi tiết đơn hàng")
        df_cart = pd.DataFrame(cart)
        df_cart["Tổng"] = df_cart["price"] * df_cart["quantity"]
        st.table(df_cart[["item_name", "quantity", "price", "Tổng"]].style.format({"price": "{:,}", "Tổng": "{:,}"}))

        subtotal = df_cart["Tổng"].sum()
        vat = subtotal * 0.1
        total = subtotal + vat

        st.markdown(f"**Tạm tính:** {subtotal:,}₫")
        st.markdown(f"**VAT (10%):** {vat:,}₫")
        st.markdown(f"### 💰 Tổng cộng: {total:,}₫")

        st.divider()
        name = st.text_input("👤 Họ và tên người mua:")
        account = st.text_input("🏦 Số tài khoản ngân hàng:")

        if st.button("✅ Xác nhận thanh toán"):
            if not name or not account:
                st.warning("⚠️ Vui lòng điền đầy đủ thông tin.")
            else:
                st.success(f"🎉 Thanh toán thành công cho {name} (STK: {account})!")
                st.balloons()
                st.session_state.cart = []
                st.session_state.page = "home"
                st.rerun()

        if st.button("↩️ Quay lại trang chính"):
            st.session_state.page = "home"
            st.rerun()
