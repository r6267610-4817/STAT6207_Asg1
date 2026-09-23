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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# 1. Page Configuration & Title
# ---------------------------------------------------------
st.set_page_config(
    page_title="STAT6207 Image Retrieval & KNN Classifier",
    page_icon="🐱🐶",
    layout="wide"
)

st.title("🐱🐶 Image Retrieval & KNN Classifier Web System")
st.markdown(
    "This system demonstrates feature extraction using pre-trained models, "
    "image distance metrics (L1, L2, Cosine), and KNN classifier performance "
    "evaluation (STAT6207 Coursework)."
)

# ---------------------------------------------------------
# 2. Feature Extractor Loading & Caching
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
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    extractor.to(device)
    extractor.eval()
    return extractor, device


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_image_embedding(image, extractor, device):
    tensor_img = transform(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = extractor(tensor_img)
        feat = torch.flatten(feat, 1)
    return feat.cpu().numpy().flatten()


# ---------------------------------------------------------
# 3. Distance Metrics
# ---------------------------------------------------------
def compute_distances(feature_matrix, query_feat, metric):
    """Compute distance from query vector to all rows of feature_matrix."""
    if metric == 'euclidean':
        return np.linalg.norm(feature_matrix - query_feat, axis=1)
    elif metric == 'manhattan':
        return np.sum(np.abs(feature_matrix - query_feat), axis=1)
    elif metric == 'cosine':
        dot = np.dot(feature_matrix, query_feat)
        norms = np.linalg.norm(feature_matrix, axis=1) * np.linalg.norm(query_feat)
        return 1.0 - (dot / (norms + 1e-9))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------
# 4. Sidebar Controls
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

k_neighbors = st.sidebar.slider(
    "Number of Neighbors (K)", min_value=1, max_value=15, value=5
)

# ---------------------------------------------------------
# 5. Load & Build Dataset Feature Cache
# ---------------------------------------------------------
dataset_dir = 'dataset'


@st.cache_data(show_spinner="Extracting features from dataset...")
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
            except Exception as e:
                print(f"Skipping {p}: {e}")

    return paths, np.array(features), np.array(labels)


# Build cached dataset + load extractor once
image_paths, feature_matrix, label_array = build_dataset_cache(selected_model_name)
extractor, device = load_feature_extractor(selected_model_name)

# Check dataset loaded
if len(image_paths) == 0:
    st.error("❌ No images found in `dataset/cat/` and `dataset/dog/`. "
             "Please make sure the dataset is in place.")
    st.stop()

st.sidebar.success(f"✅ Loaded {len(image_paths)} images "
                   f"({sum(label_array == 0)} cats, {sum(label_array == 1)} dogs)")

# ---------------------------------------------------------
# 6. Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Task 2: Image Similarity Retrieval",
    "🤖 Task 3 & 4: KNN Prediction & Failed Cases",
    "📊 Advance: Model & Metric Comparison",
    "📄 About & Report"
])

# ---------------------------------------------------------
# TAB 1: Image Similarity Retrieval (Basic Task 2)
# ---------------------------------------------------------
with tab1:
    st.header("Top-5 Similar & Dissimilar Image Retrieval")

    uploaded_file = st.file_uploader(
        "Upload a Query Image", type=["jpg", "jpeg", "png"], key="query"
    )

    if uploaded_file is not None:
        query_img = Image.open(uploaded_file)
        col_q, _ = st.columns([1, 4])
        with col_q:
            st.image(query_img, caption="Query Image", use_container_width=True)

        query_feat = get_image_embedding(query_img, extractor, device)
        distances = compute_distances(feature_matrix, query_feat, selected_metric)

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

        # Optional heatmap
        if st.checkbox("Show pairwise distance heatmap (sample of 20 images)"):
            import matplotlib.pyplot as plt
            import seaborn as sns

            sample_size = min(20, len(feature_matrix))
            sample_idx = np.random.choice(len(feature_matrix), sample_size, replace=False)
            sample_feats = feature_matrix[sample_idx]

            dist_mat = np.zeros((sample_size, sample_size))
            for i in range(sample_size):
                for j in range(sample_size):
                    dist_mat[i, j] = compute_distances(
                        sample_feats[j:j + 1], sample_feats[i], selected_metric
                    )[0]

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(dist_mat, cmap='viridis', ax=ax)
            ax.set_title(f"Pairwise Distance Matrix ({selected_metric})")
            st.pyplot(fig)

# ---------------------------------------------------------
# TAB 2: KNN Classification + Failed Cases (Basic Task 3 & 4)
# ---------------------------------------------------------
with tab2:
    st.header("KNN Classifier Prediction & Failed Cases")

    # ---- Part A: Live prediction on uploaded image ----
    st.subheader("🧪 Part A: Predict on Uploaded Image")
    knn = KNeighborsClassifier(n_neighbors=k_neighbors, metric=selected_metric)
    knn.fit(feature_matrix, label_array)

    test_file = st.file_uploader(
        "Upload an Unseen Image for Classification",
        type=["jpg", "jpeg", "png"],
        key="knn_test"
    )

    if test_file is not None:
        test_img = Image.open(test_file)
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.image(test_img, caption="Test Image", use_container_width=True)
        with col_b:
            test_feat = get_image_embedding(test_img, extractor, device).reshape(1, -1)
            pred_label = knn.predict(test_feat)[0]
            class_names = {0: "Cat 🐱", 1: "Dog 🐶"}
            st.success(f"**KNN Prediction: {class_names[pred_label]}**")

    # ---- Part B: Fixed 10-image unseen test set with failed cases ----
    st.markdown("---")
    st.subheader("📉 Part B: Failed Cases on 10 Unseen Images")

    # Use a fixed split for reproducibility
    X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
        feature_matrix,
        label_array,
        np.array(image_paths),
        test_size=10,
        stratify=label_array,
        random_state=42
    )

    knn_eval = KNeighborsClassifier(n_neighbors=k_neighbors, metric=selected_metric)
    knn_eval.fit(X_tr, y_tr)
    y_pred = knn_eval.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    num_failed = int(np.sum(y_te != y_pred))

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc * 100:.1f}%")
    col2.metric("Correct", f"{10 - num_failed}/10")
    col3.metric("Failed", f"{num_failed}/10")

    class_names_short = {0: "Cat", 1: "Dog"}
    cols = st.columns(5)
    for i in range(len(p_te)):
        with cols[i % 5]:
            st.image(p_te[i], use_container_width=True)
            true_lbl = class_names_short[y_te[i]]
            pred_lbl = class_names_short[y_pred[i]]
            if y_te[i] == y_pred[i]:
                st.caption(f"✅ True: {true_lbl} | Pred: {pred_lbl}")
            else:
                st.caption(f"❌ **FAILED**  \nTrue: {true_lbl} | Pred: {pred_lbl}")

# ---------------------------------------------------------
# TAB 3: Advance Performance Comparison Matrix
# ---------------------------------------------------------
with tab3:
    st.header("Advance Tasks: 3 Models × 3 Metrics Performance Matrix")

    if os.path.exists('advance_results.csv'):
        df_res = pd.read_csv('advance_results.csv')

        st.subheader("📋 Detailed Accuracy Breakdown")
        st.dataframe(df_res, use_container_width=True)

        pivot_df = df_res.pivot(
            index='Encoding Model',
            columns='Distance Metric',
            values='KNN Accuracy (%)'
        )
        st.subheader("📈 Accuracy Comparison Matrix (%)")
        st.table(pivot_df)

        st.subheader("📊 Performance Visual Comparison")
        st.bar_chart(pivot_df)

        st.subheader("🔍 Key Findings")
        st.markdown("""
        - **EfficientNet-B0** achieves consistent **97.5%** accuracy across all metrics.
        - **VGG16** performs best with **L1 (Manhattan)** and **Cosine** (97.5%),
          but drops to **95.0%** with **L2 (Euclidean)** — consistent with the
          curse of dimensionality in high-dimensional feature spaces.
        - **Cosine** distance is the most universally robust metric across all encoders.
        """)
    else:
        st.warning(
            "`advance_results.csv` not found! Please run the notebook "
            "`Asg1.ipynb` first to generate experiment results."
        )

# ---------------------------------------------------------
# TAB 4: About & Report
# ---------------------------------------------------------
with tab4:
    st.header("About This Project")
    st.markdown("""
    **STAT6207 Applied Deep Learning — Assignment 1**

    **Student:** [TONG, Ka Lui]  
    **Student ID:** [1155253729]  
    **Date:** 23 September 2026

    ---

    ### 🎯 Task Coverage

    | Task | Description | Status |
    |------|-------------|--------|
    | Basic 1 | Download 100+ cat & 100+ dog images, encode with ResNet18 | ✅ |
    | Basic 2 | Top-5 most similar / dissimilar retrieval | ✅ |
    | Basic 3 | Build KNN classifier | ✅ |
    | Basic 4 | Visualize 10 unseen predictions & failed cases | ✅ |
    | Basic 5 | Build a website (this app) | ✅ |
    | Advance 1 | Compare 3 encoding models | ✅ |
    | Advance 2 | Compare 3 distance metrics (L1, L2, Cosine) | ✅ |
    | Advance 3 | Analyze strengths / weaknesses | ✅ |

    ---

    ### 🧪 Methodology

    - **Dataset:** 200 images (100 cats + 100 dogs)
    - **Encoding Models:** ResNet18, VGG16, EfficientNet-B0 (pre-trained on ImageNet)
    - **Distance Metrics:** L1 (Manhattan), L2 (Euclidean), Cosine
    - **Classifier:** K-Nearest Neighbors (K=5 default)
    - **Split:** 80/20 stratified train/test

    ---

    ### 🔬 Analysis Summary

    **Why EfficientNet-B0 is most consistent:**  
    Its 1,280-dim features sit between ResNet18 (512) and VGG16 (4,096), avoiding
    both the under-capacity of ResNet18 and the curse of dimensionality of VGG16.
    Compound scaling (depth + width + resolution) yields generalizable features.

    **Why L2 fails for VGG16:**  
    VGG16's 4,096-dim features are sparse in high-dimensional space. L2 distance
    amplifies large coordinate differences, making neighbors less discriminative.
    L1 is more robust because it doesn't square the differences.

    **Why Cosine is universally strong:**  
    Cosine measures directional alignment and ignores magnitude. It is
    scale-independent, which explains why it performs well across all encoders.

    ---

    ### 🛠 Tools Used

    - **OpenCode** (agentic coder) for code generation
    - **PyTorch** for feature extraction
    - **scikit-learn** for KNN and metrics
    - **Streamlit** for this web interface

    ### 📎 OpenCode Session Transcript

    See the submitted session transcript file (`.md` or `.txt`) for full prompts
    and AI responses.
    """)
