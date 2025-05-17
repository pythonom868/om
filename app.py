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
import re
import json
import csv
import xlsxwriter
import requests
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
from kneed import KneeLocator
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import difflib

# Download NLTK resources (wrapped in try-except to handle offline scenarios)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    pass  # Will handle this gracefully during runtime

# Directory paths for model persistence
MODEL_DIR = "models"
DATA_DIR = "training_data"
EXPORT_DIR = "exports"
SUMMARY_DIR = "summaries"

# Create directories if they don't exist
for directory in [MODEL_DIR, DATA_DIR, EXPORT_DIR, SUMMARY_DIR]:
    os.makedirs(directory, exist_ok=True)

# Paths for model files
NB_MODEL_PATH = os.path.join(MODEL_DIR, "nb_model.pkl")
RNN_MODEL_PATH = os.path.join(MODEL_DIR, "rnn_model.keras")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
SUMMARIZER_PATH = os.path.join(MODEL_DIR, "summarizer.pkl")

# Import TensorFlow components
Tokenizer = tf.keras.preprocessing.text.Tokenizer
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

# ----------- Helper Functions -----------

def get_tesseract_path():
    """Handle Tesseract path based on platform"""
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

# ----------- Error Handling -----------

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass

def safe_process(func):
    """Decorator for safer processing with more detailed error messages"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            st.error(error_msg)
            return None
    return wrapper

# ----------- ML Functions -----------

def train_naive_bayes(X, y):
    """Train a Naive Bayes classifier"""
    model = make_pipeline(CountVectorizer(max_features=5000), MultinomialNB())
    model.fit(X, y)
    return model

def create_rnn_model(input_length, num_classes, embedding_dim=128):
    """Create an RNN model with improved architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_dim, input_length=input_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ----------- Text Extraction -----------

@safe_process
def extract_text_from_image(image, lang='eng'):
    """Extract text from an image with language support"""
    return pytesseract.image_to_string(image, lang=lang)

@safe_process
def extract_text_from_pdf(pdf_file, lang='eng'):
    """Extract text from PDF with language support"""
    images = convert_from_path(pdf_file)
    return "\n".join(extract_text_from_image(image, lang=lang) for image in images)

# ----------- Text Analysis -----------

def preprocess_text(text):
    """Preprocess text for NLP tasks"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text, top_n=10):
    """Extract keywords from text using TF-IDF"""
    try:
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
        # If text is too short, return empty list
        if len(text.split()) < 5:
            return []
        
        # Fit transform on the text
        response = tfidf.fit_transform([text])
        
        # Get feature names and scores
        feature_names = tfidf.get_feature_names_out()
        scores = response.toarray()[0]
        
        # Sort by scores
        keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N keywords
        return keywords[:top_n]
    except Exception as e:
        st.error(f"Error extracting keywords: {e}")
        return []

def generate_summary(text, max_sentences=5):
    """Generate a summary using extractive summarization"""
    if not text or len(text) < 100:
        return text
    
    try:
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Clean sentences
        clean_sentences = [preprocess_text(sentence) for sentence in sentences]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        sentence_vectors = vectorizer.fit_transform(clean_sentences)
        
        # Calculate sentence scores based on TF-IDF sum
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = np.sum(sentence_vectors[i].toarray())
            sentence_scores.append((i, score))
        
        # Sort sentences by score and take top N
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sentence_scores[:max_sentences]]
        
        # Sort indices to maintain original order
        top_indices.sort()
        
        # Create summary
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return text[:500] + "..." if len(text) > 500 else text

def compare_documents(text1, text2):
    """Compare two documents and find similarities"""
    # Clean texts
    clean_text1 = preprocess_text(text1)
    clean_text2 = preprocess_text(text2)
    
    # Get similarity ratio
    similarity = difflib.SequenceMatcher(None, clean_text1, clean_text2).ratio()
    
    # Use TF-IDF for content comparison
    tfidf = TfidfVectorizer()
    try:
        tfidf_matrix = tfidf.fit_transform([clean_text1, clean_text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0
    
    # Find common phrases
    tokens1 = set(clean_text1.split())
    tokens2 = set(clean_text2.split())
    common_words = tokens1.intersection(tokens2)
    
    return {
        "similarity_ratio": similarity,
        "cosine_similarity": cosine_sim,
        "common_word_count": len(common_words),
        "common_words": list(common_words)[:20]  # Limit to 20 words
    }

# ----------- File Processing -----------

def process_files(file_list, language='eng'):
    """Process files and extract text with language support"""
    dataset = []
    all_text = ""
    classification_counts = {}
    individual_texts = {}  # Store text for each file separately
    summaries = {}  # Store summaries for each file
    keywords = {}  # Store keywords for each file
    
    for filepath in file_list:
        filename = os.path.basename(filepath)
        # Use the filename without extension as the label
        label = os.path.splitext(filename)[0]
        
        try:
            text = ""
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(filepath, lang=language)
            else:
                text = extract_text_from_image(Image.open(filepath), lang=language)
            
            # Generate summary and keywords if text is extracted
            if text.strip():
                summary = generate_summary(text)
                doc_keywords = extract_keywords(text)
                
                classification_counts[label] = classification_counts.get(label, 0) + 1
                dataset.append({
                    "filename": filename,
                    "text": text.strip(),
                    "label": label,
                    "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "language": language,
                    "char_count": len(text),
                    "word_count": len(text.split())
                })
                
                # Store individual text for each file
                individual_texts[filename] = text.strip()
                
                # Store summary and keywords
                summaries[filename] = summary
                keywords[filename] = doc_keywords
                
                # Add to combined text
                all_text += f"\n\n--- {filename} ---\n{text.strip()[:500]}...\n"
            else:
                st.warning(f"⚠️ No text extracted from {filename}")
        except Exception as e:
            st.error(f"❌ Failed to process {filename}: {e}")
    
    return dataset, all_text, classification_counts, individual_texts, summaries, keywords

def get_files_from_upload(uploaded_files):
    """Process uploaded files"""
    file_list = []
    for file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.read())
        file_list.append(temp_path)
    return file_list

def get_files_from_zip(uploaded_zip):
    """Extract files from a zip archive"""
    file_list = []
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, uploaded_zip.name)
        
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
            
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.bmp')):
                    file_list.append(os.path.join(root, file))
    return file_list

def get_files_from_folder(folder_path):
    """Get files from a local folder"""
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.bmp')):
                file_list.append(os.path.join(root, file))
    return file_list

# ----------- Model Management -----------

def save_model(model, filename):
    """Save model to file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return False
    
def load_model(filename):
    """Load model from file with error handling"""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model {filename}: {e}")
    return None

def save_training_data(df):
    """Save training data and merge with existing"""
    try:
        if os.path.exists(TRAINING_DATA_PATH):
            existing_df = pd.read_csv(TRAINING_DATA_PATH)
            # Concatenate and remove duplicates based on filename
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['filename'])
            combined_df.to_csv(TRAINING_DATA_PATH, index=False)
        else:
            df.to_csv(TRAINING_DATA_PATH, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving training data: {e}")
        return False

def load_training_data():
    """Load saved training data if exists"""
    try:
        if os.path.exists(TRAINING_DATA_PATH):
            return pd.read_csv(TRAINING_DATA_PATH)
    except Exception as e:
        st.error(f"Error loading training data: {e}")
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
    texts = df["text"].fillna("").tolist()
    
    with st.spinner("Training Naive Bayes model..."):
        try:
            nb_model = train_naive_bayes(texts, y)
        except Exception as e:
            st.error(f"Error training Naive Bayes model: {e}")
            return None, None, None, None, None
    
    with st.spinner("Preparing RNN data..."):
        try:
            tokenizer = Tokenizer(num_words=10000)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            max_length = min(max([len(seq) for seq in sequences]) if sequences else 100, 500)
            padded_seq = pad_sequences(sequences, maxlen=max_length)
        except Exception as e:
            st.error(f"Error preparing RNN data: {e}")
            return nb_model, None, None, label_map, unique_labels
    
    with st.spinner("Training RNN model..."):
        try:
            rnn_model = create_rnn_model(max_length, len(unique_labels))
            # Add early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=3, restore_best_weights=True
            )
            rnn_model.fit(
                padded_seq, 
                y, 
                epochs=5, 
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
        except Exception as e:
            st.error(f"Error training RNN model: {e}")
            return nb_model, None, tokenizer, label_map, unique_labels
    
    # Save the models if auto_save is True
    if auto_save:
        save_model(nb_model, NB_MODEL_PATH)
        
        if rnn_model:
            try:
                rnn_model.save(RNN_MODEL_PATH)
            except Exception as e:
                st.warning(f"Could not save RNN model: {e}")
        
        metadata = {
            'tokenizer': tokenizer,
            'label_map': label_map,
            'unique_labels': unique_labels,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_samples': len(df),
            'max_sequence_length': max_length
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

def predict_with_models(df, nb_model, rnn_model, tokenizer, label_map, unique_labels, metadata=None):
    """Make predictions using trained models with improved error handling"""
    texts = df["text"].fillna("").tolist()
    result_df = df.copy()
    
    # Get max sequence length from metadata if available
    max_length = 100
    if metadata and 'max_sequence_length' in metadata:
        max_length = metadata['max_sequence_length']
    
    # Naive Bayes prediction
    try:
        nb_preds = nb_model.predict(texts)
        nb_labels = [unique_labels[pred] for pred in nb_preds]
        result_df["NaiveBayes_Label"] = nb_labels
    except Exception as e:
        st.error(f"Error in Naive Bayes prediction: {e}")
        result_df["NaiveBayes_Label"] = ["Error" for _ in range(len(texts))]
    
    # RNN prediction
    if rnn_model and tokenizer:
        try:
            sequences = tokenizer.texts_to_sequences(texts)
            padded_seq = pad_sequences(sequences, maxlen=max_length)
            rnn_preds = rnn_model.predict(padded_seq)
            rnn_labels = [unique_labels[np.argmax(pred)] for pred in rnn_preds]
            
            # Add confidence scores for RNN predictions
            confidence_scores = np.max(rnn_preds, axis=1)
            
            result_df["RNN_Label"] = rnn_labels
            result_df["RNN_Confidence"] = [f"{score:.2%}" for score in confidence_scores]
            
            # Create combined prediction field
            result_df["Prediction"] = [
                f"NB: {nb} | RNN: {rnn} ({conf})"
                for nb, rnn, conf in zip(
                    result_df["NaiveBayes_Label"], 
                    rnn_labels, 
                    [f"{score:.2%}" for score in confidence_scores]
                )
            ]
        except Exception as e:
            st.error(f"Error in RNN prediction: {e}")
            result_df["RNN_Label"] = ["Error" for _ in range(len(texts))]
            result_df["RNN_Confidence"] = ["N/A" for _ in range(len(texts))]
            result_df["Prediction"] = [
                f"NB: {nb} | RNN: Error"
                for nb in result_df["NaiveBayes_Label"]
            ]
    else:
        # If RNN model is not available, use only Naive Bayes
        result_df["RNN_Label"] = ["N/A" for _ in range(len(texts))]
        result_df["RNN_Confidence"] = ["N/A" for _ in range(len(texts))]
        result_df["Prediction"] = [f"NB: {nb} | RNN: N/A" for nb in result_df["NaiveBayes_Label"]]
    
    return result_df

def load_saved_models():
    """Load all saved models if they exist"""
    nb_model = load_model(NB_MODEL_PATH)
    metadata = load_model(METADATA_PATH)
    rnn_model = None
    
    if os.path.exists(RNN_MODEL_PATH):
        try:
            rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading RNN model: {e}")
    
    if nb_model and metadata:
        return nb_model, rnn_model, metadata.get('tokenizer'), metadata.get('label_map'), metadata.get('unique_labels'), metadata
    
    return None, None, None, None, None, None

# ----------- Document Search Function -----------

def search_documents(documents, query, threshold=0.2):
    """Search through documents for a specific query"""
    if not query.strip() or not documents:
        return []
    
    results = []
    query = query.lower()
    
    for filename, text in documents.items():
        if not text:
            continue
            
        # Simple search first
        if query in text.lower():
            # Calculate relevance score based on frequency and position
            frequency = text.lower().count(query)
            position = text.lower().find(query) / len(text) if len(text) > 0 else 1
            score = (frequency * 0.7) + ((1 - position) * 0.3)
            
            # Get context around the query
            idx = text.lower().find(query)
            start = max(0, idx - 100)
            end = min(len(text), idx + len(query) + 100)
            context = text[start:end]
            
            # Highlight the query in context
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
                
            results.append({
                "filename": filename,
                "score": min(score, 1.0),  # Cap at 1.0
                "context": context,
                "match_type": "direct"
            })
        else:
            # For documents without direct matches, use TF-IDF similarity
            try:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([query, text.lower()])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                
                if similarity > threshold:
                    results.append({
                        "filename": filename,
                        "score": similarity,
                        "context": text[:200] + "..." if len(text) > 200 else text,
                        "match_type": "semantic"
                    })
            except Exception:
                # If vectorization fails (e.g., empty text), skip
                continue
    
    # Sort results by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

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
    except Exception as e:
        st.warning(f"Error finding optimal clusters: {e}")
        optimal_k = min(3, max_k)  # Fallback to 3 clusters if KneeLocator fails
    
    return optimal_k, wcss

def create_vectorizer(df):
    """Create and fit TF-IDF vectorizer on document text"""
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words='english',
        min_df=1,
        max_df=0.9
    )
    vectorized_data = vectorizer.fit_transform(df["text"].fillna(""))
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
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=min(30, vectorized_data.shape[0]-1))
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
            # Try to use the same dimensionality reduction on centers
            centers = kmeans.cluster_centers_
            reducer = PCA(n_components=2)
            centers_2d = reducer.fit_transform(centers)
            
            fig.add_scatter(
                x=centers_2d[:, 0],
                y=centers_2d[:, 1],
                mode='markers',
                marker=dict(color='black', size=15, symbol='x'),
                name='Cluster Centers'
            )
    except Exception as e:
        pass
    
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
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        )
    )
    
    return fig

def get_top_cluster_terms(vectorizer, kmeans, n_terms=10):
    """Get top terms characterizing each cluster"""
    # Get cluster centers
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_terms = {}
    for i in range(kmeans.n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :n_terms]]
        cluster_terms[i] = top_terms
    
    return cluster_terms

def analyze_clusters(df, kmeans):
    """Provide analysis of document clusters"""
    df_analysis = df.copy()
    df_analysis['Cluster'] = kmeans.labels_
    
    cluster_stats = {}
    for cluster_id in range(kmeans.n_clusters):
        cluster_docs = df_analysis[df_analysis['Cluster'] == cluster_id]
        
        if len(cluster_docs) > 0:
            avg_length = cluster_docs['text'].apply(len).mean()
            common_labels = cluster_docs['label'].value_counts().head(3).to_dict()
            
            cluster_stats[cluster_id] = {
                'document_count': len(cluster_docs),
                'avg_document_length': int(avg_length),
                'most_common_labels': common_labels,
                'sample_documents': cluster_docs['filename'].head(5).tolist()
            }
    
    return cluster_stats

# ----------- Export Functions -----------

def export_to_csv(df, filename="document_analysis.csv"):
    """Export analysis results to CSV"""
    filepath = os.path.join(EXPORT_DIR, filename)
    df.to_csv(filepath, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return filepath

def export_to_excel(df, filename="document_analysis.xlsx"):
    """Export analysis results to Excel with formatting"""
    filepath = os.path.join(EXPORT_DIR, filename)
    
    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Analysis Results', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Analysis Results']
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply header format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Adjust column widths
        for i, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).apply(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(i, i, min(max_len, 50))
    
    return filepath

def export_json(data, filename="document_analysis.json"):
    """Export analysis results to JSON"""
    filepath = os.path.join(EXPORT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filepath

def export_summaries(summaries, filename="document_summaries.txt"):
    """Export document summaries to text file"""
    filepath = os.path.join(SUMMARY_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for doc_name, summary in summaries.items():
            f.write(f"=== {doc_name} ===\n")
            f.write(f"{summary}\n\n")
    
    return filepath

# ----------- Streamlit UI -----------

def main():
    st.set_page_config(
        page_title="Document Analyzer",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e6f0ff;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        gap: 4px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c78a8 !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #4c78a8;
        color: white;
        border-radius: 4px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App title
    st.title("📄 Document Analyzer")
    st.markdown("""
    Upload documents (PDF, images) for automated text extraction, analysis, and classification.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ("Upload Files", "Upload ZIP", "Use Local Folder")
        )
        
        # OCR Language selection
        ocr_language = st.selectbox(
            "OCR Language:",
            ['eng', 'fra', 'deu', 'spa', 'ita', 'por', 'nld', 'chi_sim', 'chi_tra', 'jpn', 'kor'],
            index=0
        )
        
        # Processing options
        st.subheader("Processing Options")
        
        enable_summarization = st.checkbox("Generate summaries", value=True)
        enable_keyword_extraction = st.checkbox("Extract keywords", value=True)
        enable_classification = st.checkbox("Document classification", value=True)
        enable_clustering = st.checkbox("Document clustering", value=True)
        
        if enable_clustering:
            cluster_method = st.selectbox(
                "Dimension Reduction Method:",
                ("pca", "svd", "tsne", "umap"),
                index=0
            )
            
            auto_clusters = st.checkbox("Auto-detect clusters", value=True)
            if not auto_clusters:
                num_clusters = st.slider("Number of clusters", 2, 10, 3)
        
        # Advanced options
        with st.expander("Advanced Options"):
            summary_length = st.slider("Max summary sentences", 1, 15, 5)
            keyword_count = st.slider("Keywords to extract", 5, 30, 10)
            save_training_checkbox = st.checkbox("Save data for training", value=True)
        
        st.subheader("Models")
        model_status = "No models loaded"
        
        nb_model, rnn_model, tokenizer, label_map, unique_labels, metadata = load_saved_models()
        
        if nb_model is not None:
            model_status = "Models loaded successfully"
            if metadata and 'last_updated' in metadata:
                model_status += f"\nLast updated: {metadata['last_updated']}"
            if metadata and 'num_samples' in metadata:
                model_status += f"\nTrained on: {metadata['num_samples']} samples"
        
        st.info(model_status)
        
        if st.button("Train/Update Models"):
            df = load_training_data()
            if len(df) > 0:
                nb_model, rnn_model, tokenizer, label_map, unique_labels, metadata = train_models(df)
                st.success("Models trained successfully!")
            else:
                st.warning("No training data available")
    
    # Main content area with tabs
    tabs = st.tabs(["Document Processing", "Search & Compare", "Analysis & Visualization", "Export"])
    
    with tabs[0]:
        st.header("Document Processing")
        
        # File input based on selected method
        uploaded_files = None
        uploaded_zip = None
        folder_path = None
        
        if input_method == "Upload Files":
            uploaded_files = st.file_uploader(
                "Upload PDF or image files",
                type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                file_list = get_files_from_upload(uploaded_files)
                st.info(f"Uploaded {len(file_list)} files")
        
        elif input_method == "Upload ZIP":
            uploaded_zip = st.file_uploader(
                "Upload ZIP file containing documents",
                type=["zip"]
            )
            
            if uploaded_zip:
                file_list = get_files_from_zip(uploaded_zip)
                st.info(f"Extracted {len(file_list)} files from ZIP")
        
        elif input_method == "Use Local Folder":
            folder_path = st.text_input("Enter folder path on server")
            
            if folder_path and os.path.isdir(folder_path):
                file_list = get_files_from_folder(folder_path)
                st.info(f"Found {len(file_list)} files in folder")
            elif folder_path:
                st.error("Invalid folder path")
                file_list = []
            else:
                file_list = []
        
        # Process button
        process_button = st.button("Process Documents")
        
        if process_button and 'file_list' in locals() and file_list:
            with st.spinner("Processing documents..."):
                # Process files
                dataset, all_text, classification_counts, individual_texts, summaries, keywords = process_files(
                    file_list, language=ocr_language
                )
                
                if not dataset:
                    st.error("No text could be extracted from the documents.")
                else:
                    # Convert to DataFrame
                    df = pd.DataFrame(dataset)
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.individual_texts = individual_texts
                    st.session_state.summaries = summaries
                    st.session_state.keywords = keywords
                    
                    # Save training data if option is checked
                    if save_training_checkbox:
                        save_training_data(df)
                    
                    # Show success message
                    st.success(f"✅ Processed {len(df)} documents successfully!")
                    
                    # Display statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Document Statistics")
                        st.write(f"Total documents: {len(df)}")
                        st.write(f"Total text extracted: {df['char_count'].sum()} characters")
                        st.write(f"Average document length: {int(df['char_count'].mean())} characters")
                    
                    with col2:
                        st.subheader("Document Labels")
                        for label, count in classification_counts.items():
                            st.write(f"{label}: {count} documents")
                    
                    # Classify documents if option is checked and models are available
                    if enable_classification and nb_model is not None:
                        with st.spinner("Classifying documents..."):
                            try:
                                # Make predictions
                                result_df = predict_with_models(
                                    df, nb_model, rnn_model, tokenizer, label_map, unique_labels, metadata
                                )
                                
                                # Store results
                                st.session_state.result_df = result_df
                                
                                # Show results
                                st.subheader("Classification Results")
                                st.dataframe(
                                    result_df[["filename", "label", "NaiveBayes_Label", "RNN_Label", "RNN_Confidence"]]
                                )
                            except Exception as e:
                                st.error(f"Error classifying documents: {e}")
                    
                    # Perform clustering if option is checked
                    if enable_clustering and len(df) >= 3:
                        with st.spinner("Clustering documents..."):
                            try:
                                # Create vectorizer
                                vectorizer, vectorized_data = create_vectorizer(df)
                                
                                # Determine number of clusters
                                if auto_clusters:
                                    n_clusters, wcss = get_optimal_clusters(vectorized_data, max_k=min(10, len(df)-1))
                                    st.info(f"Auto-detected optimal clusters: {n_clusters}")
                                else:
                                    n_clusters = num_clusters
                                
                                # Train KMeans
                                kmeans = train_kmeans(vectorized_data, n_clusters=n_clusters)
                                
                                # Store in session state
                                st.session_state.kmeans = kmeans
                                st.session_state.vectorizer = vectorizer
                                st.session_state.vectorized_data = vectorized_data
                                
                                # Add cluster info to results
                                if hasattr(st.session_state, 'result_df'):
                                    st.session_state.result_df['Cluster'] = kmeans.labels_
                                
                                # Show basic clustering results
                                df_cluster = df.copy()
                                df_cluster['Cluster'] = kmeans.labels_
                                cluster_counts = df_cluster['Cluster'].value_counts()
                                
                                st.subheader("Clustering Results")
                                for cluster_id, count in cluster_counts.items():
                                    st.write(f"Cluster {cluster_id}: {count} documents")
                            except Exception as e:
                                st.error(f"Error clustering documents: {e}")
                    
                    # Display extracted text samples
                    with st.expander("View Extracted Text Samples"):
                        for i, (filename, text) in enumerate(list(individual_texts.items())[:3]):
                            st.markdown(f"**{filename}**")
                            st.text_area(f"Extracted text", text[:500] + "..." if len(text) > 500 else text, height=150)
                            st.markdown("---")
                    
                    # Display summaries if option is checked
                    if enable_summarization:
                        with st.expander("View Document Summaries"):
                            for filename, summary in summaries.items():
                                st.markdown(f"**{filename}**")
                                st.write(summary)
                                st.markdown("---")
                    
                    # Display keywords if option is checked
                    if enable_keyword_extraction:
                        with st.expander("View Extracted Keywords"):
                            for filename, kw_list in keywords.items():
                                st.markdown(f"**{filename}**")
                                if kw_list:
                                    for keyword, score in kw_list:
                                        st.write(f"- {keyword} ({score:.3f})")
                                else:
                                    st.write("No keywords extracted")
                                st.markdown("---")
        
        elif process_button:
            st.warning("Please upload or select files first")
    
    # Search & Compare tab
    with tabs[1]:
        st.header("Search & Compare Documents")
        
        if hasattr(st.session_state, 'individual_texts') and st.session_state.individual_texts:
            # Search feature
            st.subheader("Search Documents")
            search_query = st.text_input("Enter search term")
            
            if search_query:
                search_results = search_documents(st.session_state.individual_texts, search_query)
                
                if search_results:
                    st.write(f"Found {len(search_results)} results")
                    
                    for result in search_results:
                        with st.expander(f"{result['filename']} (Score: {result['score']:.2f})"):
                            st.markdown(f"**Match type:** {result['match_type']}")
                            st.markdown(f"**Context:**")
                            st.markdown(result['context'].replace(search_query, f"**{search_query}**"))
                else:
                    st.info("No results found")
            
            # Document comparison
            st.subheader("Compare Documents")
            
            document_options = list(st.session_state.individual_texts.keys())
            if len(document_options) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    doc1 = st.selectbox("First Document", document_options, index=0)
                
                with col2:
                    remaining_options = [doc for doc in document_options if doc != doc1]
                    doc2 = st.selectbox("Second Document", remaining_options, index=0)
                
                if st.button("Compare Documents"):
                    with st.spinner("Comparing documents..."):
                        text1 = st.session_state.individual_texts[doc1]
                        text2 = st.session_state.individual_texts[doc2]
                        
                        comparison_results = compare_documents(text1, text2)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Similarity Ratio", f"{comparison_results['similarity_ratio']:.2f}")
                        
                        with col2:
                            st.metric("Cosine Similarity", f"{comparison_results['cosine_similarity']:.2f}")
                        
                        with col3:
                            st.metric("Common Words", comparison_results['common_word_count'])
                        
                        st.markdown("### Common Terms")
                        st.write(", ".join(comparison_results['common_words']))
            else:
                st.info("Upload at least two documents to compare")
        else:
            st.info("Process documents first to use search and comparison features")
    
    # Analysis & Visualization tab
    with tabs[2]:
        st.header("Analysis & Visualization")
        
        if hasattr(st.session_state, 'vectorized_data') and hasattr(st.session_state, 'kmeans'):
            # Clustering visualization
            st.subheader("Document Clustering")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 2D visualization
                reduced_data = reduce_dimensions(
                    st.session_state.vectorized_data, 
                    method=cluster_method,
                    n_components=2
                )
                fig_2d = plot_clusters_2d(st.session_state.df, reduced_data, st.session_state.kmeans)
                st.plotly_chart(fig_2d, use_container_width=True)
            
            with col2:
                # 3D visualization
                reduced_data_3d = reduce_dimensions(
                    st.session_state.vectorized_data, 
                    method=cluster_method,
                    n_components=3
                )
                fig_3d = plot_clusters_3d(st.session_state.df, reduced_data_3d, st.session_state.kmeans)
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # Cluster analysis
            st.subheader("Cluster Analysis")
            
            # Get top terms per cluster
            top_terms = get_top_cluster_terms(st.session_state.vectorizer, st.session_state.kmeans)
            
            # Get cluster statistics
            cluster_stats = analyze_clusters(st.session_state.df, st.session_state.kmeans)
            
            # Display cluster details
            for cluster_id in range(st.session_state.kmeans.n_clusters):
                with st.expander(f"Cluster {cluster_id} ({cluster_stats[cluster_id]['document_count']} documents)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Top terms:**")
                        st.write(", ".join(top_terms[cluster_id]))
                        
                        st.write("**Most common labels:**")
                        for label, count in cluster_stats[cluster_id]['most_common_labels'].items():
                            st.write(f"- {label}: {count}")
                    
                    with col2:
                        st.write("**Sample documents:**")
                        for doc in cluster_stats[cluster_id]['sample_documents']:
                            st.write(f"- {doc}")
                        
                        st.write(f"**Avg. document length:** {cluster_stats[cluster_id]['avg_document_length']} chars")
            
            # Elbow curve visualization
            if 'auto_clusters' in locals() and auto_clusters:
                st.subheader("Cluster Optimization")
                n_clusters, wcss = get_optimal_clusters(st.session_state.vectorized_data)
                fig_elbow = plot_elbow_curve(wcss)
                st.pyplot(fig_elbow)
        
        elif hasattr(st.session_state, 'df'):
            # Basic document statistics
            st.subheader("Document Statistics")
            
            # Length distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(st.session_state.df['char_count'], bins=20, alpha=0.7)
            ax.set_xlabel('Document Length (characters)')
            ax.set_ylabel('Count')
            ax.set_title('Document Length Distribution')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Document count by label
            if 'label' in st.session_state.df.columns:
                label_counts = st.session_state.df['label'].value_counts()
                
                fig = px.bar(
                    x=label_counts.index,
                    y=label_counts.values,
                    labels={'x': 'Document Label', 'y': 'Count'},
                    title='Document Count by Label',
                    color=label_counts.values,
                    color_continuous_scale=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig)
        else:
            st.info("Process documents first to view analysis and visualizations")
    
    # Export tab
    with tabs[3]:
        st.header("Export Results")
        
        if hasattr(st.session_state, 'df'):
            export_format = st.radio(
                "Export format:",
                ("CSV", "Excel", "JSON", "Text Summaries")
            )
            
            export_button = st.button("Export Data")
            
            if export_button:
                with st.spinner("Exporting data..."):
                    try:
                        # Different export based on format
                        if export_format == "CSV":
                            # Export analysis results to CSV
                            if hasattr(st.session_state, 'result_df'):
                                filepath = export_to_csv(st.session_state.result_df)
                            else:
                                filepath = export_to_csv(st.session_state.df)
                        
                        elif export_format == "Excel":
                            # Export analysis results to Excel
                            if hasattr(st.session_state, 'result_df'):
                                filepath = export_to_excel(st.session_state.result_df)
                            else:
                                filepath = export_to_excel(st.session_state.df)
                        
                        elif export_format == "JSON":
                            # Export as JSON
                            export_data = {
                                "documents": st.session_state.df.to_dict(orient='records')
                            }
                            
                            # Add predictions if available
                            if hasattr(st.session_state, 'result_df'):
                                export_data["predictions"] = st.session_state.result_df.to_dict(orient='records')
                            
                            # Add clustering if available
                            if hasattr(st.session_state, 'kmeans'):
                                export_data["clusters"] = analyze_clusters(
                                    st.session_state.df, st.session_state.kmeans
                                )
                            
                            filepath = export_json(export_data)
                        
                        elif export_format == "Text Summaries":
                            # Export summaries as text
                            if hasattr(st.session_state, 'summaries'):
                                filepath = export_summaries(st.session_state.summaries)
                            else:
                                st.error("No summaries available")
                                filepath = None
                        
                        if filepath:
                            st.success(f"✅ Exported successfully to: {filepath}")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
        else:
            st.info("Process documents first to export results")

# Run the application
if __name__ == "__main__":
    main()
