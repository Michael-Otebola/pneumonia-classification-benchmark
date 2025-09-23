import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from scipy import ndimage
import matplotlib.cm as cm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AI Chest X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem 3rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.8);
        font-size: 1.2rem;
        text-align: center;
        margin: 0;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);
    }
    
    .normal-prediction {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 8px 20px rgba(79, 172, 254, 0.3);
    }
    
    .pneumonia-prediction {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        box-shadow: 0 8px 20px rgba(250, 112, 154, 0.3);
    }
    
    .prediction-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .confidence-score {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    /* Upload area styling */
    .upload-container {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Model Setup
# ---------------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load the state dict to inspect its structure
        state_dict = torch.load("results/ResNet50.pth", map_location=device)
        
        # Determine the number of classes from the saved model
        if 'fc.0.weight' in state_dict:
            # Sequential fc layer
            num_classes = state_dict['fc.0.weight'].shape[0]
            model = models.resnet50(weights=None)
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, num_classes)
            )
        elif 'fc.weight' in state_dict:
            # Single Linear fc layer
            num_classes = state_dict['fc.weight'].shape[0]
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            # Fallback to 2 classes
            num_classes = 2
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Update classes list based on detected number of classes
        if num_classes == 1:
            classes = ["PNEUMONIA"]  # Binary classification with single output
        else:
            classes = ["NORMAL", "PNEUMONIA"]
        
        return model, device, classes, num_classes
        
    except Exception as e:
        st.error(f"🔍 Detailed error: {str(e)}")
        raise e

# ---------------------------
# Grad-CAM Implementation
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = dict(model.named_modules())[target_layer]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)
        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8
        return heatmap

# ---------------------------
# Helper Functions
# ---------------------------
def create_confidence_chart(confidence_scores, labels):
    """Create a modern confidence chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=confidence_scores,
            marker_color=['#4facfe', '#fa709a'] if len(labels) > 1 else ['#fa709a'],
            text=[f'{score:.1f}%' for score in confidence_scores],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        title_x=0.5,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12),
        margin=dict(l=20, r=20, t=60, b=20),
        height=300,
    )
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.1)', range=[0, 100])
    
    return fig

def process_gradcam_overlay(img, cam):
    """Create a sophisticated Grad-CAM overlay"""
    # Resize cam to match image dimensions using PIL
    cam_img = Image.fromarray((cam * 255).astype(np.uint8))
    cam_resized = cam_img.resize(img.size, Image.BILINEAR)
    cam_resized = np.array(cam_resized) / 255.0
    
    # Apply colormap using matplotlib
    heatmap = cm.jet(cam_resized)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Convert original image to numpy array
    img_cv = np.array(img)
    if img_cv.ndim == 2:  # grayscale
        img_cv = np.stack([img_cv] * 3, axis=-1)  # Convert to RGB
    
    # Create overlay
    overlay = np.uint8(heatmap * 0.4 + img_cv * 0.6)
    return overlay

# ---------------------------
# Main App Interface
# ---------------------------
def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🫁 AI Chest X-ray Analysis</h1>
        <p class="header-subtitle">Advanced Pneumonia Detection with Explainable AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    try:
        model, device, classes, num_classes = load_model()
        model_loaded = True
        
        # Display model info in sidebar
        st.sidebar.success(f"✅ Model loaded successfully!")
        st.sidebar.info(f"📊 Number of classes: {num_classes}")
        st.sidebar.info(f"🏷️ Classes: {', '.join(classes)}")
        
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {str(e)}")
        model_loaded = False
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Analysis Settings")
        
        # Model info
        st.markdown("""
        <div class="info-card">
            <h4>🤖 Model Information</h4>
            <p><strong>Architecture:</strong> ResNet-50</p>
            <p><strong>Classes:</strong> Normal, Pneumonia</p>
            <p><strong>Device:</strong> {}</p>
        </div>
        """.format("GPU" if torch.cuda.is_available() else "CPU"), unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        <div class="info-card">
            <h4>📋 Instructions</h4>
            <ol>
                <li>Upload a chest X-ray image</li>
                <li>Wait for AI analysis</li>
                <li>Review prediction & heatmap</li>
                <li>Interpret the results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div class="info-card">
            <h4>⚠️ Medical Disclaimer</h4>
            <p><small>This tool is for educational purposes only. Always consult healthcare professionals for medical diagnosis.</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload X-ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear chest X-ray image in JPG, JPEG, or PNG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📷 Uploaded X-ray Image", use_column_width=True)
            
            # Image info
            st.markdown(f"""
            <div class="metric-card">
                <strong>Image Details</strong><br>
                Size: {img.size[0]} × {img.size[1]} pixels<br>
                Format: {img.format}<br>
                Mode: {img.mode}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🔍 AI Analysis Results")
            
            # Show processing
            with st.spinner("🤖 AI is analyzing the X-ray..."):
                # Preprocessing
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                # Prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    
                    if num_classes == 1:
                        # Single output - binary classification with sigmoid
                        probability = torch.sigmoid(outputs).item()
                        pred_class = 1 if probability > 0.5 else 0
                        
                        # For display purposes, show both normal and pneumonia probabilities
                        normal_prob = (1 - probability) * 100
                        pneumonia_prob = probability * 100
                        confidence_scores = [normal_prob, pneumonia_prob]
                        
                        prediction = "PNEUMONIA" if probability > 0.5 else "NORMAL"
                        main_confidence = pneumonia_prob if prediction == "PNEUMONIA" else normal_prob
                        
                    else:
                        # Multi-class classification with softmax
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence_scores = (probabilities * 100).cpu().numpy()[0]
                        _, pred = torch.max(outputs, 1)
                        prediction = classes[pred.item()]
                        main_confidence = confidence_scores[pred.item()]
                
                # Simulate processing time for better UX
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                progress_bar.empty()
            
            # Display prediction
            prediction_class = "normal-prediction" if prediction == "NORMAL" else "pneumonia-prediction"
            
            st.markdown(f"""
            <div class="prediction-card {prediction_class}">
                <div class="prediction-title">Diagnosis: {prediction}</div>
                <div class="confidence-score">{main_confidence:.1f}% Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence chart - always show both Normal and Pneumonia
            chart_labels = ['Normal', 'Pneumonia']
            fig = create_confidence_chart(confidence_scores, chart_labels)
            st.plotly_chart(fig, use_container_width=True)
    
    # Grad-CAM Analysis (Full Width)
    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 🎯 Explainable AI - Attention Heatmap")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.markdown("#### 🔥 Grad-CAM Heatmap")
            with st.spinner("Generating attention heatmap..."):
                # Generate Grad-CAM
                grad_cam = GradCAM(model, target_layer="layer4")
                pred_idx = 0 if num_classes == 1 else (pred.item() if num_classes > 1 else 0)
                cam = grad_cam.generate(input_tensor, class_idx=pred_idx)
                overlay = process_gradcam_overlay(img, cam)
                
                st.image(overlay, caption="🎯 AI Attention Areas", use_container_width=True)
        
        with col4:
            st.markdown("#### 📊 Heatmap Analysis")
            
            # Heatmap statistics
            cam_stats = {
                "Max Attention": f"{np.max(cam):.3f}",
                "Mean Attention": f"{np.mean(cam):.3f}",
                "Attention Variance": f"{np.var(cam):.3f}",
                "Focus Areas": "Lung regions" if np.max(cam) > 0.7 else "Distributed"
            }
            
            for key, value in cam_stats.items():
                st.markdown(f"""
                <div class="metric-card">
                    <strong>{key}:</strong> {value}
                </div>
                """, unsafe_allow_html=True)
            
            # Interpretation guide
            st.markdown("""
            <div class="info-card">
                <h4>🔍 How to Read the Heatmap</h4>
                <ul>
                    <li><strong>Red areas:</strong> High AI attention</li>
                    <li><strong>Blue areas:</strong> Low AI attention</li>
                    <li><strong>Focus regions:</strong> Key diagnostic areas</li>
                </ul>
                <p><small>The AI focuses on areas most relevant for diagnosis.</small></p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()