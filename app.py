import os
import glob
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------
# 1. Page Configuration & Title
# ---------------------------------------------------------
st.set_page_config(
    page_title="STAT6207 Image Retrieval & KNN Classifier",
    page_icon="🐱🐶",
    layout="wide"
)

st.title("🐱🐶 Image Retrieval & KNN Classifier Web System")
st.markdown("This system demonstrates feature extraction using pre-trained models, image distance metrics (L1, L2, Cosine), and KNN classifier performance evaluation (STAT6207 Coursework).")

# ---------------------------------------------------------
# 2. Feature Extractor Loading & Caching (st.cache_resource)
# ---------------------------------------------------------
@st.cache_resource
def load_feature_extractor(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        extractor = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier = model.classifier[:-1]
        extractor = model
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        extractor = model
    extractor.to(device)
    extractor.eval()
    return extractor, device

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_image_embedding(image, extractor, device):
    tensor_img = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = extractor(tensor_img)
        feat = torch.flatten(feat, 1)
    return feat.cpu().numpy().flatten()

# ---------------------------------------------------------
# 3. Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parameter Settings")
selected_model_name = st.sidebar.selectbox(
    "Select Feature Encoding Model",
    ['resnet18', 'vgg16', 'efficientnet_b0']
)

selected_metric = st.sidebar.selectbox(
    "Select Distance Metric",
    ['euclidean', 'manhattan', 'cosine']
)

k_neighbors = st.sidebar.slider("Number of Neighbors (K)", min_value=1, max_value=15, value=5)

# ---------------------------------------------------------
# 4. Load & Build Dataset Feature Cache
# ---------------------------------------------------------
dataset_dir = 'dataset'

@st.cache_data
def build_dataset_cache(model_name):
    extractor, device = load_feature_extractor(model_name)
    paths, features, labels = [], [], []
    class_map = {'cat': 0, 'dog': 1}
    
    for cname, label in class_map.items():
        folder = os.path.join(dataset_dir, cname)
        for p in glob.glob(os.path.join(folder, '*.*')):
            try:
                img = Image.open(p)
                feat = get_image_embedding(img, extractor, device)
                paths.append(p)
                features.append(feat)
                labels.append(label)
            except:
                pass
    return paths, np.array(features), np.array(labels)

image_paths, feature_matrix, label_array = build_dataset_cache(selected_model_name)
extractor, device = load_feature_extractor(selected_model_name)

# Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Task 2: Image Similarity Retrieval", 
    "🤖 Task 3 & 4: KNN Prediction & Unseen Cases", 
    "📊 Advance: Model & Metric Comparison"
])

# ---------------------------------------------------------
# TAB 1: Image Similarity Retrieval (Basic Task 2)
# ---------------------------------------------------------
with tab1:
    st.header("Top-5 Similar & Dissimilar Image Retrieval")
    
    uploaded_file = st.file_uploader("Upload a Query Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        query_img = Image.open(uploaded_file)
        col_q, _ = st.columns([1, 2])
        with col_q:
            st.image(query_img, caption="Query Image", width=200)
            
        query_feat = get_image_embedding(query_img, extractor, device)
        
        # Calculate Distance
        if selected_metric == 'euclidean':
            distances = np.linalg.norm(feature_matrix - query_feat, axis=1)
        elif selected_metric == 'manhattan':
            distances = np.sum(np.abs(feature_matrix - query_feat), axis=1)
        elif selected_metric == 'cosine':
            dot = np.dot(feature_matrix, query_feat)
            norms = np.linalg.norm(feature_matrix, axis=1) * np.linalg.norm(query_feat)
            distances = 1 - (dot / (norms + 1e-9))
            
        sorted_idx = np.argsort(distances)
        top5_sim = sorted_idx[:5]
        top5_dissim = sorted_idx[-5:][::-1]
        
        st.subheader("🟢 Top-5 Most Similar Images")
        cols_sim = st.columns(5)
        for i, idx in enumerate(top5_sim):
            with cols_sim[i]:
                st.image(image_paths[idx], use_container_width=True)
                st.caption(f"Dist: {distances[idx]:.3f}")
                
        st.subheader("🔴 Top-5 Most Dissimilar Images")
        cols_dissim = st.columns(5)
        for i, idx in enumerate(top5_dissim):
            with cols_dissim[i]:
                st.image(image_paths[idx], use_container_width=True)
                st.caption(f"Dist: {distances[idx]:.3f}")

# ---------------------------------------------------------
# TAB 2: KNN Classification (Basic Task 3 & 4)
# ---------------------------------------------------------
with tab2:
    st.header("KNN Classifier Prediction")
    
    # Train KNN Model
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric=selected_metric)
    knn.fit(feature_matrix, label_array)
    
    test_file = st.file_uploader("Upload an Unseen Image for Classification", type=["jpg", "jpeg", "png"], key="knn_test")
    
    if test_file is not None:
        test_img = Image.open(test_file)
        st.image(test_img, caption="Test Image", width=250)
        
        test_feat = get_image_embedding(test_img, extractor, device).reshape(1, -1)
        pred_label = knn.predict(test_feat)[0]
        class_names = {0: "Cat 🐱", 1: "Dog 🐶"}
        
        st.success(f"**KNN Prediction Result: {class_names[pred_label]}**")

# ---------------------------------------------------------
# TAB 3: Advance Performance Comparison Matrix (Advance Tasks)
# ---------------------------------------------------------
with tab3:
    st.header("Advance Tasks: 3 Models x 3 Metrics Performance Matrix")
    
    if os.path.exists('advance_results.csv'):
        df_res = pd.read_csv('advance_results.csv')
        
        # Display Dataset Table
        st.subheader("📋 Detailed Accuracy Breakdown")
        st.dataframe(df_res, use_container_width=True)
        
        # Pivot Table Matrix
        pivot_df = df_res.pivot(index='Encoding Model', columns='Distance Metric', values='KNN Accuracy (%)')
        st.subheader("📈 Accuracy Comparison Matrix (%)")
        st.table(pivot_df)
        
        # Chart
        st.subheader("📊 Performance Visual Comparison")
        st.bar_chart(pivot_df)
    else:
        st.warning("`advance_results.csv` not found! Please execute `python main_assignment.py` first to generate the experiment results.")