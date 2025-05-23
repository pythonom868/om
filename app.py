import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import zipfile
import tempfile
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import io
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from kneed import KneeLocator
import json
import csv
import xlsxwriter

# Directory paths for model persistence
MODEL_DIR = "models"
DATA_DIR = "training_data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Paths for model files
NB_MODEL_PATH = os.path.join(MODEL_DIR, "nb_model.pkl")
RNN_MODEL_PATH = os.path.join(MODEL_DIR, "rnn_model.keras")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

# Handle Tesseract path based on platform
def get_tesseract_path():
    if os.name == 'nt':  # Windows
        return r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:  # Unix/Linux/MacOS
        return pytesseract.pytesseract.tesseract_cmd

# Try to set Tesseract path
try:
    pytesseract.pytesseract.tesseract_cmd = get_tesseract_path()
except Exception as e:
    # Will be handled during runtime
    pass

# ----------- ML Functions -----------

def train_naive_bayes(X, y):
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X, y)
    return model

def create_rnn_model(input_length, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=1000, output_dim=64, input_length=input_length),
        tf.keras.layers.SimpleRNN(128),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ----------- Text Extraction -----------

def extract_text_from_image(image):
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    try:
        images = convert_from_path(pdf_file)
        return "\n".join(extract_text_from_image(image) for image in images)
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""

# ----------- File Processing -----------

def process_files(file_list):
    dataset = []
    all_text = ""
    classification_counts = {}
    individual_texts = {}  # Store text for each file separately
    
    for filepath in file_list:
        filename = os.path.basename(filepath)
        # Use the filename without extension as the label
        label = os.path.splitext(filename)[0]
        
        try:
            text = ""
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
            else:
                text = extract_text_from_image(Image.open(filepath))
            
            classification_counts[label] = classification_counts.get(label, 0) + 1
            dataset.append({
                "filename": filename,
                "text": text.strip(),
                "label": label,
                "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Store individual text for each file
            individual_texts[filename] = text.strip()
            
            # Add to combined text
            all_text += f"\n--- {filename} ---\n{text.strip()[:500]}...\n"
        except Exception as e:
            st.error(f"❌ Failed to process {filename}: {e}")
    
    return dataset, all_text, classification_counts, individual_texts

def get_files_from_upload(uploaded_files):
    file_list = []
    for file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.read())
        file_list.append(temp_path)
    return file_list

def get_files_from_zip(uploaded_zip):
    file_list = []
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, uploaded_zip.name)
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                    file_list.append(os.path.join(root, file))
    return file_list

def get_files_from_folder(folder_path):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                file_list.append(os.path.join(root, file))
    return file_list

# ----------- Model Management -----------

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
def load_model(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def save_training_data(df):
    """Save new training data and merge with existing"""
    if os.path.exists(TRAINING_DATA_PATH):
        existing_df = pd.read_csv(TRAINING_DATA_PATH)
        # Concatenate and remove duplicates based on filename
        combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['filename'])
        combined_df.to_csv(TRAINING_DATA_PATH, index=False)
    else:
        df.to_csv(TRAINING_DATA_PATH, index=False)

def load_training_data():
    """Load saved training data if exists"""
    if os.path.exists(TRAINING_DATA_PATH):
        return pd.read_csv(TRAINING_DATA_PATH)
    return pd.DataFrame(columns=["filename", "text", "label", "processed_date"])

# ----------- Model Training and Prediction -----------

def train_models(df, auto_save=True):
    """Train models with the provided dataframe"""
    # Check if we have enough data
    if len(df) < 2 or len(df["label"].unique()) < 2:
        st.warning("⚠️ Not enough data or unique labels to train model")
        return None, None, None, None, None
    
    unique_labels = df["label"].unique()
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = df["label"].map(label_map)
    texts = df["text"].tolist()
    
    with st.spinner("Training Naive Bayes model..."):
        nb_model = train_naive_bayes(texts, y)
    
    with st.spinner("Preparing RNN data..."):
        tokenizer = Tokenizer(num_words=1000)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_seq = pad_sequences(sequences, maxlen=100)
    
    with st.spinner("Training RNN model..."):
        rnn_model = create_rnn_model(100, len(unique_labels))
        rnn_model.fit(padded_seq, y, epochs=3, verbose=0)
    
    # Save the models if auto_save is True
    if auto_save:
        save_model(nb_model, NB_MODEL_PATH)
        rnn_model.save(RNN_MODEL_PATH)
        metadata = {
            'tokenizer': tokenizer,
            'label_map': label_map,
            'unique_labels': unique_labels,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_samples': len(df)
        }
        save_model(metadata, METADATA_PATH)
    
    return nb_model, rnn_model, tokenizer, label_map, unique_labels

def update_model_with_new_data(new_df):
    """Update existing model with new data"""
    # Get existing training data
    training_df = load_training_data()
    
    # Add new data
    combined_df = pd.concat([training_df, new_df]).drop_duplicates(subset=['filename'])
    
    # Save combined training data
    combined_df.to_csv(TRAINING_DATA_PATH, index=False)
    
    # Train with combined data
    return train_models(combined_df)

def predict_with_models(df, nb_model, rnn_model, tokenizer, label_map, unique_labels):
    """Make predictions using trained models"""
    texts = df["text"].tolist()
    
    # Naive Bayes prediction
    nb_preds = nb_model.predict(texts)
    nb_labels = [unique_labels[pred] for pred in nb_preds]
    
    # RNN prediction
    sequences = tokenizer.texts_to_sequences(texts)
    padded_seq = pad_sequences(sequences, maxlen=100)
    rnn_preds = rnn_model.predict(padded_seq)
    rnn_labels = [unique_labels[np.argmax(pred)] for pred in rnn_preds]
    
    # Add confidence scores for RNN predictions
    confidence_scores = np.max(rnn_preds, axis=1)
    
    # Store original predictions for possible debugging
    df["NaiveBayes_Label"] = nb_labels
    df["RNN_Label"] = rnn_labels
    df["RNN_Confidence"] = [f"{score:.2%}" for score in confidence_scores]
    
    # Create combined prediction field
    df["Prediction"] = [
        f"NB: {nb} | RNN: {rnn} ({conf})"
        for nb, rnn, conf in zip(nb_labels, rnn_labels, [f"{score:.2%}" for score in confidence_scores])
    ]
    
    return df

def load_saved_models():
    """Load saved models if they exist"""
    nb_model = load_model(NB_MODEL_PATH)
    
    metadata = load_model(METADATA_PATH)
    rnn_model = None
    
    if os.path.exists(RNN_MODEL_PATH):
        try:
            rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading RNN model: {e}")
    
    if nb_model and rnn_model and metadata:
        return nb_model, rnn_model, metadata['tokenizer'], metadata['label_map'], metadata['unique_labels']
    
    return None, None, None, None, None

# ----------- K-Means Clustering Functions -----------

def get_optimal_clusters(vectorized_data, max_k=10):
    """Find optimal number of clusters using Elbow method"""
    # Make sure max_k doesn't exceed number of samples
    n_samples = vectorized_data.shape[0]
    max_k = min(max_k, n_samples - 1)  # Ensure max_k is at most n_samples-1
    
    wcss = []  # Within-cluster sum of squares
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(vectorized_data)
        wcss.append(kmeans.inertia_)
    
    # Find elbow point
    try:
        kl = KneeLocator(range(1, max_k + 1), wcss, curve="convex", direction="decreasing")
        optimal_k = kl.elbow if kl.elbow else min(3, max_k)  # Default to 3 or max_k if smaller
    except:
        optimal_k = min(3, max_k)  # Fallback to 3 clusters if KneeLocator fails
    
    return optimal_k, wcss

def create_vectorizer(df):
    """Create and fit TF-IDF vectorizer on document text"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        min_df=2,
        max_df=0.85
    )
    vectorized_data = vectorizer.fit_transform(df["text"])
    save_model(vectorizer, VECTORIZER_PATH)
    return vectorizer, vectorized_data

def train_kmeans(vectorized_data, n_clusters=3):
    """Train K-means clustering model"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(vectorized_data)
    save_model(kmeans, KMEANS_MODEL_PATH)
    return kmeans

def reduce_dimensions(vectorized_data, method='pca', n_components=2, random_state=42):
    """Reduce dimensions of vectorized data for visualization"""
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(vectorized_data.toarray())
    elif method == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(vectorized_data)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(vectorized_data.toarray())
    elif method == 'umap':
        # Import UMAP only if needed
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            return reducer.fit_transform(vectorized_data.toarray())
        except ImportError:
            st.error("UMAP is not installed. Install with 'pip install umap-learn'")
            # Fallback to PCA if UMAP is not available
            reducer = PCA(n_components=n_components, random_state=random_state)
            return reducer.fit_transform(vectorized_data.toarray())
    else:
        raise ValueError("Unsupported dimension reduction method")

def plot_elbow_curve(wcss):
    """Plot elbow curve to help find optimal K"""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(range(1, len(wcss) + 1), wcss, 'bo-')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Square)')
    plt.grid(True)
    return fig

def plot_clusters_2d(df, reduced_data, kmeans):
    """Plot 2D visualization of clusters"""
    df_plot = df.copy()
    df_plot['Cluster'] = kmeans.labels_
    
    fig = px.scatter(
        x=reduced_data[:, 0], 
        y=reduced_data[:, 1],
        color=kmeans.labels_,
        hover_name=df_plot['filename'],
        title="Document Clusters (2D View)",
        labels={'color': 'Cluster'},
        color_continuous_scale=px.colors.qualitative.G10
    )
    
    # Add cluster centers if possible
    try:
        if hasattr(kmeans, 'cluster_centers_'):
            # Need to transform the cluster centers if we've reduced dimensions
            centers = kmeans.cluster_centers_
            if reduced_data.shape[1] == 2 and centers.shape[1] != 2:
                # This is simplified and may not work for all dimensionality reduction methods
                # For actual implementation, you'd need to apply the same transformation to centers
                pass
            else:
                pca = PCA(n_components=2)
                centers_2d = pca.fit_transform(centers)
                fig.add_scatter(
                    x=centers_2d[:, 0],
                    y=centers_2d[:, 1],
                    mode='markers',
                    marker=dict(color='black', size=15, symbol='x'),
                    name='Cluster Centers'
                )
    except Exception as e:
        st.warning(f"Could not plot cluster centers: {e}")
    
    return fig

def plot_clusters_3d(df, reduced_data_3d, kmeans):
    """Plot 3D visualization of clusters"""
    df_plot = df.copy()
    df_plot['Cluster'] = kmeans.labels_
    
    fig = px.scatter_3d(
        x=reduced_data_3d[:, 0], 
        y=reduced_data_3d[:, 1], 
        z=reduced_data_3d[:, 2],
        color=kmeans.labels_,
        hover_name=df_plot['filename'],
        title="Document Clusters (3D View)",
        labels={'color': 'Cluster'},
        color_continuous_scale=px.colors.qualitative.G10
    )
    
    fig.update_layout(scene=dict(
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        zaxis_title='Dimension 3'
    ))
    
    return fig

def plot_document_similarity_heatmap(df, vectorized_data):
    """Generate a document similarity heatmap"""
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(vectorized_data)
    
    # Create heatmap
    fig = px.imshow(
        similarity_matrix,
        labels=dict(x="Document Index", y="Document Index", color="Similarity"),
        x=df['filename'],
        y=df['filename'],
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        title="Document Similarity Heatmap",
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    
    return fig

def analyze_clusters(df, kmeans):
    """Analyze cluster contents and extract key features"""
    df_analysis = df.copy()
    df_analysis['Cluster'] = kmeans.labels_
    
    # Group by cluster and count documents per cluster
    cluster_counts = df_analysis['Cluster'].value_counts().sort_index()
    
    # Get document counts per label within each cluster
    cluster_label_distribution = df_analysis.groupby(['Cluster', 'label']).size().unstack(fill_value=0)
    
    return cluster_counts, cluster_label_distribution

def visualize_clusters(df):
    """Main function to create and visualize clusters from document data"""
    if len(df) < 3:
        st.warning("⚠️ Need at least 3 documents for meaningful clustering")
        return

    # Create document vectors
    with st.spinner("Creating document vectors..."):
        vectorizer, vectorized_data = create_vectorizer(df)
    
    # Find optimal number of clusters
    with st.spinner("Finding optimal number of clusters..."):
        optimal_k, wcss = get_optimal_clusters(vectorized_data)
        
        # Elbow curve plot
        st.subheader("📈 Elbow Method for Optimal Number of Clusters")
        st.write(f"Suggested optimal number of clusters: **{optimal_k}**")
        elbow_fig = plot_elbow_curve(wcss)
        st.pyplot(elbow_fig)
    
    # Allow user to select number of clusters
    selected_k = st.slider("Select number of clusters:", min_value=2, max_value=10, value=optimal_k)
    
    # Train K-means with selected k
    with st.spinner(f"Clustering documents into {selected_k} groups..."):
        kmeans = train_kmeans(vectorized_data, n_clusters=selected_k)
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs([
        "2D Visualization", 
        "3D Visualization", 
        "Document Similarity", 
        "Cluster Analysis"
    ])
    
    # Tab 1: 2D Visualization
    with viz_tabs[0]:
        st.subheader("🔍 2D Cluster Visualization")
        
        dimension_method = st.selectbox(
            "Select dimension reduction method for 2D:",
            options=["PCA", "t-SNE", "SVD", "UMAP"],
            index=0
        )
        
        method_map = {
            "PCA": "pca",
            "t-SNE": "tsne",
            "SVD": "svd",
            "UMAP": "umap"
        }
        
        # Reduce to 2D and plot
        with st.spinner(f"Generating 2D visualization using {dimension_method}..."):
            reduced_data_2d = reduce_dimensions(
                vectorized_data, 
                method=method_map[dimension_method],
                n_components=2
            )
            cluster_fig_2d = plot_clusters_2d(df, reduced_data_2d, kmeans)
            st.plotly_chart(cluster_fig_2d, use_container_width=True, key="plot_2d_cluster")
    
    # Tab 2: 3D Visualization
    with viz_tabs[1]:
        st.subheader("🧊 3D Cluster Visualization")
        
        dimension_method_3d = st.selectbox(
            "Select dimension reduction method for 3D:",
            options=["PCA", "t-SNE", "SVD", "UMAP"],
            index=0,
            key="3d_method"
        )
        
        # Reduce to 3D and plot
        with st.spinner(f"Generating 3D visualization using {dimension_method_3d}..."):
            reduced_data_3d = reduce_dimensions(
                vectorized_data, 
                method=method_map[dimension_method_3d],
                n_components=3
            )
            cluster_fig_3d = plot_clusters_3d(df, reduced_data_3d, kmeans)
            st.plotly_chart(cluster_fig_3d, use_container_width=True, key="plot_3d_cluster")
    
    # Tab 3: Document Similarity
    with viz_tabs[2]:
        st.subheader("📊 Document Similarity Heatmap")
        
        if len(df) > 50:
            st.warning("⚠️ Heatmap may be crowded with more than 50 documents")
        
        with st.spinner("Generating similarity heatmap..."):
            similarity_fig = plot_document_similarity_heatmap(df, vectorized_data)
            st.plotly_chart(similarity_fig, use_container_width=True, key="plot_similarity_heatmap")
    
    # Tab 4: Cluster Analysis
    with viz_tabs[3]:
        st.subheader("📑 Cluster Analysis")
        
        # Get cluster statistics
        cluster_counts, cluster_label_distribution = analyze_clusters(df, kmeans)
        
        # Display cluster statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Documents per Cluster:**")
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                labels={'x': 'Cluster', 'y': 'Number of Documents'},
                text=cluster_counts.values
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True, key="plot_cluster_counts")
        
        with col2:
            st.write("**Label Distribution per Cluster:**")
            if not cluster_label_distribution.empty:
                fig = px.bar(
                    cluster_label_distribution,
                    labels={'value': 'Count', 'variable': 'Document Type'}
                )
                st.plotly_chart(fig, use_container_width=True, key="plot_label_distribution")
            else:
                st.info("No label distribution data available")
        
        # Show documents in each cluster
        st.write("**Documents in Each Cluster:**")
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = kmeans.labels_
        
        selected_cluster = st.selectbox(
            "Select cluster to view documents:",
            options=sorted(df_with_clusters['Cluster'].unique())
        )
        
        cluster_docs = df_with_clusters[df_with_clusters['Cluster'] == selected_cluster]
        st.write(f"**Cluster {selected_cluster}** contains {len(cluster_docs)} documents:")
        st.dataframe(cluster_docs[['filename', 'label']])
    
    # Return cluster assignments for potential further use
    df['cluster'] = kmeans.labels_
    return df

def save_extracted_text(text, filename, format_type):
    """Save extracted text in various formats (TXT, CSV, Excel)"""
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == "txt":
        # Save as TXT file
        output_path = f"{base_filename}_{timestamp}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        return output_path
    
    elif format_type == "csv":
        # Save as CSV file with each line as a separate row
        output_path = f"{base_filename}_{timestamp}.csv"
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Add header row
            writer.writerow(["Line Number", "Text Content"])
            # Split text by lines and write each line as a row
            for i, line in enumerate(text.split('\n')):
                if line.strip():  # Skip empty lines
                    writer.writerow([i+1, line])
        return output_path
    
    elif format_type == "excel":
        # Save as Excel file with each line as a separate row
        output_path = f"{base_filename}_{timestamp}.xlsx"
        workbook = xlsxwriter.Workbook(output_path)
        worksheet = workbook.add_worksheet("Extracted Text")
        
        # Add headers with some formatting
        bold_format = workbook.add_format({'bold': True, 'bg_color': '#D9D9D9'})
        worksheet.write(0, 0, "Line Number", bold_format)
        worksheet.write(0, 1, "Text Content", bold_format)
        
        # Split text by lines and write each line as a row
        row = 1
        for i, line in enumerate(text.split('\n')):
            if line.strip():  # Skip empty lines
                worksheet.write(row, 0, i+1)
                worksheet.write(row, 1, line)
                row += 1
        
        # Adjust column widths
        worksheet.set_column(0, 0, 15)
        worksheet.set_column(1, 1, 100)
        
        workbook.close()
        return output_path
    
    return None

def batch_save_documents(file_names, individual_texts, batch_format, save_option):
    """Process batch saving of documents in various formats"""
    saved_files = []
    
    if save_option == "Individual files":
        # Save each document as separate file
        for filename in file_names:
            text = individual_texts.get(filename, "")
            if text.strip():
                saved_path = save_extracted_text(text, filename, batch_format)
                if saved_path:
                    saved_files.append(saved_path)
        
        if saved_files:
            st.success(f"✅ Saved {len(saved_files)} files successfully!")
    else:
        # Save all documents as one combined file
        combined_text = ""
        for filename in file_names:
            text = individual_texts.get(filename, "")
            if text.strip():
                combined_text += f"\n\n--- {filename} ---\n{text}\n"
        
        if combined_text.strip():
            saved_path = save_extracted_text(combined_text, "combined_documents", batch_format)
            if saved_path:
                st.success(f"✅ Combined file saved as {saved_path}")
        else:
            st.error("⚠️ No text available to save")
    
    return saved_files

# ----------- Streamlit App -----------

def main():
    st.set_page_config(page_title="📄 Document Classification Engine", page_icon="📁", layout="wide")
    st.title("📁 Intelligent Document Classification")
    # Initialize session state
    if 'nb_model' not in st.session_state:
        st.session_state['nb_model'] = None
    if 'rnn_model' not in st.session_state:
        st.session_state['rnn_model'] = None
    if 'tokenizer' not in st.session_state:
        st.session_state['tokenizer'] = None
    if 'label_map' not in st.session_state:
        st.session_state['label_map'] = None
    if 'unique_labels' not in st.session_state:
        st.session_state['unique_labels'] = None
    if 'individual_texts' not in st.session_state:
        st.session_state['individual_texts'] = {}
    if 'dataset' not in st.session_state:
        st.session_state['dataset'] = []
    if 'classification_counts' not in st.session_state:
        st.session_state['classification_counts'] = {}
    if 'processed_data' not in st.session_state:
        st.session_state['processed_data'] = None

    # Create tabs for different functions
    tabs = st.tabs([
        "📤 Upload & Extract", 
        "🏷️ Train & Classify", 
        "📊 Visualize", 
        "📋 Manage Data"
    ])
# 📷 Camera Scanner Section
    st.subheader("📷 Document Scanner")
    if st.button("Open Camera Scanner"):
        camera_image = st.camera_input("Capture Image")

        if camera_image is not None:
            image = Image.open(camera_image)
            st.image(image, caption="Captured Image", use_column_width=True)

            with st.spinner("Extracting text..."):
                extracted_text = pytesseract.image_to_string(image)
                st.success("✅ Text extracted successfully!")

            st.text_area("📝 Extracted Text", extracted_text, height=250) 

    # Tab 1: Upload & Extract
    with tabs[0]:
        st.header("📤 Upload Documents & Extract Text")
        
        upload_col1, upload_col2 = st.columns(2)
        
        with upload_col1:
            st.subheader("1️⃣ Choose Upload Method")
            upload_method = st.radio(
                "Select how to upload files:",
                options=["Upload individual files", "Upload ZIP file", "Use local folder"],
                index=0
            )
            
            file_list = []
            
            if upload_method == "Upload individual files":
                uploaded_files = st.file_uploader(
                    "Upload images or PDFs", 
                    type=["jpg", "jpeg", "png", "pdf"], 
                    accept_multiple_files=True
                )
                if uploaded_files:
                    file_list = get_files_from_upload(uploaded_files)
                    
            elif upload_method == "Upload ZIP file":
                uploaded_zip = st.file_uploader("Upload ZIP file containing images/PDFs", type=["zip"])
                if uploaded_zip:
                    file_list = get_files_from_zip(uploaded_zip)
                    
            elif upload_method == "Use local folder":
                folder_path = st.text_input("Enter path to folder containing images/PDFs")
                if folder_path and os.path.isdir(folder_path):
                    file_list = get_files_from_folder(folder_path)
                    if not file_list:
                        st.warning("⚠️ No supported files found in this folder")
        
        with upload_col2:
            st.subheader("2️⃣ Process Files")
            
            if file_list:
                st.write(f"📄 Found {len(file_list)} file(s)")
                
                if st.button("🚀 Process Files", key="process_button"):
                    with st.spinner("Processing files..."):
                        dataset, all_text, classification_counts, individual_texts = process_files(file_list)
                        
                        # Save to session state
                        st.session_state['dataset'] = dataset
                        st.session_state['classification_counts'] = classification_counts
                        st.session_state['individual_texts'] = individual_texts
                        
                        # Create DataFrame
                        processed_df = pd.DataFrame(dataset)
                        st.session_state['processed_data'] = processed_df
                        
                        # Display message
                        st.success(f"✅ Successfully processed {len(dataset)} files!")
            else:
                st.info("👆 Upload files to begin")
        
        # Show results if data is available
        if st.session_state['processed_data'] is not None:
            st.subheader("3️⃣ Extracted Data")
            
            # Display a summary and class distribution
            st.write("**Document Class Distribution:**")
            class_counts = st.session_state['classification_counts']
            
            # Create a horizontal bar chart
            if class_counts:
                fig = px.bar(
                    x=list(class_counts.values()), 
                    y=list(class_counts.keys()),
                    orientation='h',
                    labels={'x': 'Number of Documents', 'y': 'Document Class'},
                    title='Document Distribution by Class'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Allow viewing individual documents
            st.write("**Preview Extracted Text:**")
            file_names = list(st.session_state['individual_texts'].keys())
            if file_names:
                selected_file = st.selectbox("Select a file to view:", file_names)
                
                # Show the extracted text
                text = st.session_state['individual_texts'].get(selected_file, "")
                if text.strip():
                    st.text_area("Extracted Text:", value=text, height=300)
                else:
                    st.warning("⚠️ No text extracted from this file")
                
                # Save options
                st.write("**Save Options:**")
                save_cols = st.columns(3)
                with save_cols[0]:
                    save_format = st.selectbox(
                        "Save format:", 
                        options=["txt", "csv", "excel"],
                        index=0
                    )
                with save_cols[1]:
                    if st.button("💾 Save This Document"):
                        saved_path = save_extracted_text(text, selected_file, save_format)
                        if saved_path:
                            st.success(f"✅ Saved to {saved_path}")
                
                # Batch save option
                st.write("**Batch Save Options:**")
                batch_cols = st.columns([2, 1, 1])
                with batch_cols[0]:
                    batch_format = st.selectbox(
                        "Batch save format:", 
                        options=["txt", "csv", "excel"],
                        index=0,
                        key="batch_format"
                    )
                with batch_cols[1]:
                    save_option = st.radio(
                        "Save as:",
                        options=["Individual files", "One combined file"],
                        key="batch_save_option"
                    )
                with batch_cols[2]:
                    if st.button("💾 Batch Save All"):
                        batch_save_documents(
                            file_names, 
                            st.session_state['individual_texts'], 
                            batch_format, 
                            save_option
                        )

    # Tab 2: Train & Classify
    with tabs[1]:
        st.header("🏷️ Train & Classify Documents")
        
        if st.session_state['processed_data'] is None:
            st.info("👈 First process files in the Upload & Extract tab")
        else:
            train_col1, train_col2 = st.columns(2)
            
            with train_col1:
                st.subheader("1️⃣ Training Data")
                
                # Display the data to be used for training
                st.write(f"**Available Data:** {len(st.session_state['processed_data'])} documents")
                st.dataframe(
                    st.session_state['processed_data'][['filename', 'label']]
                )
                
                # Training options
                st.subheader("2️⃣ Train Models")
                
                if st.button("🧠 Train Models"):
                    with st.spinner("Training models..."):
                        # Train both models
                        nb_model, rnn_model, tokenizer, label_map, unique_labels = train_models(
                            st.session_state['processed_data']
                        )
                        
                        # Save to session state
                        st.session_state['nb_model'] = nb_model
                        st.session_state['rnn_model'] = rnn_model
                        st.session_state['tokenizer'] = tokenizer
                        st.session_state['label_map'] = label_map
                        st.session_state['unique_labels'] = unique_labels
                        
                        # Save training data for future use
                        save_training_data(st.session_state['processed_data'])
                        
                        st.success("✅ Models trained successfully!")
                
                # Option to load saved models
                if st.button("📂 Load Saved Models"):
                    with st.spinner("Loading saved models..."):
                        nb_model, rnn_model, tokenizer, label_map, unique_labels = load_saved_models()
                        
                        if nb_model and rnn_model:
                            # Save to session state
                            st.session_state['nb_model'] = nb_model
                            st.session_state['rnn_model'] = rnn_model
                            st.session_state['tokenizer'] = tokenizer
                            st.session_state['label_map'] = label_map
                            st.session_state['unique_labels'] = unique_labels
                            
                            st.success(f"✅ Models loaded successfully! Available classes: {', '.join(unique_labels)}")
                        else:
                            st.error("❌ No saved models found")
            
            with train_col2:
                st.subheader("3️⃣ Classification")
                
                # Check if models are available
                models_available = (
                    st.session_state['nb_model'] is not None and 
                    st.session_state['rnn_model'] is not None
                )
                
                if not models_available:
                    st.info("👈 First train or load models")
                else:
                    # Option to use current data or new data
                    classification_data_option = st.radio(
                        "Select data to classify:",
                        options=["Use current processed data", "Upload new files to classify"],
                        index=0
                    )
                    
                    classification_df = None
                    
                    if classification_data_option == "Use current processed data":
                        classification_df = st.session_state['processed_data'].copy()
                    else:
                        # Allow uploading new files
                        classify_files = st.file_uploader(
                            "Upload files to classify", 
                            type=["jpg", "jpeg", "png", "pdf"], 
                            accept_multiple_files=True,
                            key="classify_upload"
                        )
                        
                        if classify_files:
                            with st.spinner("Processing new files..."):
                                file_list = get_files_from_upload(classify_files)
                                dataset, _, _, _ = process_files(file_list)
                                classification_df = pd.DataFrame(dataset)
                    
                    # Run classification if data is available
                    if classification_df is not None and not classification_df.empty:
                        if st.button("🔍 Run Classification"):
                            with st.spinner("Classifying documents..."):
                                # Make predictions
                                result_df = predict_with_models(
                                    classification_df,
                                    st.session_state['nb_model'],
                                    st.session_state['rnn_model'],
                                    st.session_state['tokenizer'],
                                    st.session_state['label_map'],
                                    st.session_state['unique_labels']
                                )
                                
                                # Show results
                                st.subheader("📋 Classification Results")
                                st.dataframe(
                                    result_df[['filename', 'label', 'NaiveBayes_Label', 'RNN_Label', 'RNN_Confidence']]
                                )
                                
                                # Calculate accuracy if original labels are available
                                if 'label' in result_df.columns:
                                    nb_accuracy = (result_df['label'] == result_df['NaiveBayes_Label']).mean()
                                    rnn_accuracy = (result_df['label'] == result_df['RNN_Label']).mean()
                                    
                                    st.write("**Model Performance:**")
                                    col1, col2 = st.columns(2)
                                    col1.metric("Naive Bayes Accuracy", f"{nb_accuracy:.2%}")
                                    col2.metric("RNN Accuracy", f"{rnn_accuracy:.2%}")
                                
                                # Option to save results
                                if st.button("💾 Save Classification Results"):
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    results_path = f"classification_results_{timestamp}.csv"
                                    result_df.to_csv(results_path, index=False)
                                    st.success(f"✅ Results saved to {results_path}")

    # Tab 3: Visualize
    with tabs[2]:
        st.header("📊 Document Analysis & Visualization")
        
        if st.session_state['processed_data'] is None:
            st.info("👈 First process files in the Upload & Extract tab")
        else:
            st.subheader("Cluster Analysis")
            
            # Run clustering and visualization
            if st.button("🔍 Analyze Document Clusters"):
                with st.spinner("Analyzing document clusters..."):
                    visualize_clusters(st.session_state['processed_data'])

    # Tab 4: Manage Data
    with tabs[3]:
        st.header("📋 Data Management")
        
        data_tabs = st.tabs(["Current Data", "Saved Training Data"])
        
        # Current session data
        with data_tabs[0]:
            st.subheader("Current Session Data")
            
            if st.session_state['processed_data'] is not None:
                st.write(f"**Documents in current session:** {len(st.session_state['processed_data'])}")
                st.dataframe(st.session_state['processed_data'])
                
                # Export current data
                if st.button("💾 Export Current Data"):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    export_path = f"document_data_export_{timestamp}.csv"
                    st.session_state['processed_data'].to_csv(export_path, index=False)
                    st.success(f"✅ Data exported to {export_path}")
                
                # Clear session data
                if st.button("🗑️ Clear Current Session Data"):
                    st.session_state['processed_data'] = None
                    st.session_state['dataset'] = []
                    st.session_state['classification_counts'] = {}
                    st.session_state['individual_texts'] = {}
                    st.success("✅ Session data cleared")
                    st.experimental_rerun()
            else:
                st.info("No data in current session")
        
        # Saved training data
        with data_tabs[1]:
            st.subheader("Saved Training Data")
            
            training_df = load_training_data()
            
            if not training_df.empty:
                st.write(f"**Saved training documents:** {len(training_df)}")
                st.dataframe(training_df)
                
                # Delete saved training data
                if st.button("🗑️ Delete All Saved Training Data"):
                    if os.path.exists(TRAINING_DATA_PATH):
                        os.remove(TRAINING_DATA_PATH)
                        st.success("✅ All saved training data deleted")
                        st.experimental_rerun()
            else:
                st.info("No saved training data found")
            
            # Model management
            st.subheader("Model Management")
            
            model_files = [
                os.path.exists(NB_MODEL_PATH),
                os.path.exists(RNN_MODEL_PATH),
                os.path.exists(METADATA_PATH)
            ]
            
            if any(model_files):
                st.write("**Saved models:**")
                if model_files[0]:
                    st.write("✅ Naive Bayes model")
                if model_files[1]:
                    st.write("✅ RNN model")
                if model_files[2]:
                    # Show metadata if available
                    metadata = load_model(METADATA_PATH)
                    if metadata:
                        st.write(f"📈 Model trained on {metadata.get('num_samples', 'unknown')} samples")
                        st.write(f"🏷️ Available classes: {', '.join(metadata.get('unique_labels', []))}")
                        st.write(f"⏰ Last updated: {metadata.get('last_updated', 'unknown')}")
                
                # Delete saved models
                if st.button("🗑️ Delete All Saved Models"):
                    for model_path in [NB_MODEL_PATH, RNN_MODEL_PATH, METADATA_PATH, VECTORIZER_PATH, KMEANS_MODEL_PATH]:
                        if os.path.exists(model_path):
                            os.remove(model_path)
                    st.success("✅ All saved models deleted")
                    st.experimental_rerun()
            else:
                st.info("No saved models found")

# Run the app
if __name__ == "__main__":
    main()
