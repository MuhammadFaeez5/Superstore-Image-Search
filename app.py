import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Superstore Visual Search",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f4f7f9;
    }
    
    .query-container {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 2px dashed #dfe3e8;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .product-card {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e1e4e8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        transition: transform 0.3s ease;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .similarity-badge {
        background: linear-gradient(90deg, #28a745, #85d045);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    .product-title {
        font-size: 14px;
        font-weight: 600;
        color: #1a1a1a;
        margin-top: 10px;
        height: 40px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

@st.cache_data
def load_vectors(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, list):
        return data[0], data[1]
    return data['features'], data['paths']

model, preprocess, device = load_resources()
image_features, image_paths = load_vectors('clip_model (1).pkl')

st.title("🛒 Superstore Visual Search")
st.markdown("Search our inventory by uploading a photo. Results are filtered at **70% similarity**.")


THRESHOLD = 0.70
TOP_N = 8
IMAGE_FOLDER = "images_download" 

uploaded_file = st.file_uploader("Drop a product image here", type=["jpg", "png", "jpeg"])

if uploaded_file:
    input_img = Image.open(uploaded_file)
    
    col_query, col_results = st.columns([1, 3], gap="large")

    with col_query:
        st.subheader("Your Search")
        st.image(input_img, use_container_width=True)

    with col_results:
        st.subheader("Matching Inventory")
        
        with st.spinner("Analyzing inventory..."):
            img_tensor = preprocess(input_img).unsqueeze(0).to(device)
            with torch.no_grad():
                query_feat = model.encode_image(img_tensor)
                query_feat /= query_feat.norm(dim=-1, keepdim=True)
            
            sims = cosine_similarity(query_feat.cpu().numpy(), image_features)[0]
            top_indices = sims.argsort()[-TOP_N:][::-1]

        res_cols = st.columns(3)
        found_any = False
        
        for i, idx in enumerate(top_indices):
            score = sims[idx]
            if score < THRESHOLD:
                continue
            
            found_any = True
            file_name = os.path.basename(image_paths[idx])
            local_img_path = os.path.join(IMAGE_FOLDER, file_name)
            
            with res_cols[i % 3]:
                st.markdown(f"""
                    <div class="product-card">
                        <span class="similarity-badge">{score:.1%} Match</span>
                        <div class="product-title">{file_name}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                if os.path.exists(local_img_path):
                    st.image(local_img_path, use_container_width=True)
                else:
                    st.caption("🖼️ Image not found in local folder")
                
                st.divider()

        if not found_any:
            st.error(f"No products found. Try uploading a different image.")

else:
    st.info("Welcome! Please upload an image to see similar products from our catalog.")