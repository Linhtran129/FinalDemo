import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os
import time

# ==============================
# Load model
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
# Load data
# ==============================
@st.cache_data(show_spinner=True)
def load_data_from_drive(sample_size=5000):
    file_id = "1V4cuan8_FUBf2m-Y134P07VjvuSputh5"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    local_path = "events.json"

    if not os.path.exists(local_path):
        with st.spinner("🔽 Downloading dataset..."):
            gdown.download(url, local_path, quiet=False)

    dfs = []
    with open(local_path, "r", encoding="utf-8") as f:
        start = f.read(200).strip()
    try:
        if start.startswith("["):
            df = pd.read_json(local_path)
        else:
            for chunk in pd.read_json(local_path, lines=True, chunksize=200000):
                dfs.append(chunk)
            df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"❌ Failed to read JSON file: {e}")
        st.stop()

    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    return df

df = load_data_from_drive()
event_weight = {"view": 1, "addtocart": 3, "transaction": 5}
df["weight"] = df["event"].map(event_weight)
df["itemid"] = df["itemid"].astype(str)
df = df[df["itemid"].isin(i_id2idx.keys())]
np.random.seed(42)
price_map = {iid: np.random.randint(50, 500) * 1000 for iid in df["itemid"].unique()}
df["price"] = df["itemid"].map(price_map)

# ==============================
# Recommendation 
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
# UI Layout
# ==============================
st.set_page_config(page_title="PetShop Recommender", page_icon="🐾", layout="wide")
st.title("🐾 PetShop Product Recommender System")

# --- Sidebar ---
menu = st.sidebar.radio("📋 Choose page:", ["Home", "Cart"])

# --- Session init ---
if "cart" not in st.session_state:
    st.session_state.cart = []
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "page" not in st.session_state:
    st.session_state.page = "Home"

# --- Sync sidebar with session ---
if menu != st.session_state.page and st.session_state.page != "Checkout":
    st.session_state.page = menu

# ==============================
# Home Page
# ==============================
if st.session_state.page == "Home":
    visitor_ids = sorted(df["visitorid"].unique())
    prev_visitor = st.session_state.get("current_visitor")
    visitor_id = st.selectbox("👤 Select Visitor ID:", visitor_ids)

    if prev_visitor != visitor_id:
        st.session_state.recommendations = []
        st.session_state["current_visitor"] = visitor_id

    user_items = df[df["visitorid"] == visitor_id]["itemid"].unique()
    st.write(f"Visitor **{visitor_id}** has interacted with {len(user_items)} items.")

    if len(user_items) > 0:
        seed_item = st.selectbox("📦 Select an item to get similar recommendations:", user_items)
        item_info = df[df["itemid"] == seed_item].iloc[0]
        st.subheader("📍 Viewed item")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(item_info["image_url"], width=200)
        with col2:
            st.markdown(f"**{item_info['item_name']}**")
            st.caption(item_info["item_description"][:400])
            st.write(f"💰 Price: **{item_info['price']:,} VND**")

        if st.button("🎯 Get similar recommendations", use_container_width=True):
            rec_items = get_top_n_knn(seed_item, 10)
            if not rec_items:
                st.warning("⚠️ No similar items found. Please try another product.")
                st.session_state.recommendations = []
            else:
                rec_df = df[df["itemid"].isin(rec_items)].drop_duplicates("itemid")[
                    ["itemid", "item_name", "item_description", "image_url", "price"]
                ]
                st.session_state.recommendations = rec_df.to_dict("records")

        if st.session_state.recommendations:
            st.subheader("✨ Recommended for you")
            rec_df = pd.DataFrame(st.session_state.recommendations)
            for row_start in range(0, len(rec_df), 4):
                cols = st.columns(4)
                for i, (_, row) in enumerate(rec_df.iloc[row_start:row_start + 4].iterrows()):
                    with cols[i]:
                        st.image(row["image_url"], width=180)
                        st.markdown(f"**{row['item_name']}**")
                        st.caption(row["item_description"][:100] + "...")
                        st.write(f"💰 **{row['price']:,} VND**")
                        if st.button("🛍️ Add to cart", key=f"rec_{row['itemid']}"):
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
                            st.success(f"✅ Added {row['item_name']} to cart")

# ==============================
#  Cart Page
# ==============================
elif st.session_state.page == "Cart":
    st.header("🛒 Your Shopping Cart")

    if not st.session_state.cart:
        st.info("Your cart is empty. Go to the Home page to add items.")
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
                    q1, q2, q3 = st.columns([1, 1, 1])
                    with q1:
                        if st.button("➖", key=f"minus_{idx}") and item["quantity"] > 1:
                            item["quantity"] -= 1
                            st.rerun()
                    with q2:
                        st.markdown(f"<div style='text-align:center;font-size:18px;'><b>{item['quantity']}</b></div>", unsafe_allow_html=True)
                    with q3:
                        if st.button("➕", key=f"plus_{idx}"):
                            item["quantity"] += 1
                            st.rerun()
                with col3:
                    if st.button("🗑️ Remove", key=f"del_{idx}"):
                        st.session_state.cart.pop(idx)
                        st.rerun()
            st.divider()
            total_price += item["quantity"] * item["price"]

        st.markdown(f"### 💰 Total: **{total_price:,} VND**")
        st.divider()

        if st.button("💳 Proceed to Checkout", use_container_width=True):
            st.session_state.page = "Checkout"
            st.session_state.total_price = total_price
            st.rerun()

# ==============================
#  Checkout Page
# ==============================
elif st.session_state.page == "Checkout":
    st.title("💳 Checkout")

    if "payment_success" not in st.session_state:
        st.session_state.payment_success = False

    if not st.session_state.payment_success:
        name = st.text_input("👤 Full name")
        acc = st.text_input("🏦 Bank account number")

        if st.button("✅ Confirm Payment", use_container_width=True):
            if not name or not acc:
                st.error("⚠️ Please fill in all required fields.")
            else:
                st.session_state.payment_success = True
                st.success(f"🎉 Payment successful! Thank you, {name}, for your purchase.")
                st.balloons()
                time.sleep(3)
                st.session_state.cart = []
                st.session_state.recommendations = []
                st.rerun()

        if st.button("⬅️ Back to Cart", use_container_width=True):
            st.session_state.page = "Cart"
            st.rerun()

    else:
        st.success("🎉 Payment successful! Thank you for your purchase.")
        if st.button("🛒 Cart Page", use_container_width=True):
            st.session_state.payment_success = False
            st.session_state.page = "Cart"
            st.rerun()


