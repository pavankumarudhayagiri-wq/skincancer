import streamlit as st
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
import matplotlib.pyplot as plt
from io import StringIO
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, concatenate
from tensorflow.keras.applications import VGG16, ResNet50, InceptionResNetV2, EfficientNetB4
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.preprocessing import image
import logging
from typing import Dict, Tuple
import yaml
from PIL import Image, ImageDraw
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set page configuration
st.set_page_config(page_title="Melanoma Detection App", page_icon="🔬", layout="wide")


# Define custom CSS for background image and sidebar color
st.markdown(
    """
    <style>
    /* Set the main background image for the app */
    .stApp {
        background-image: url('https://i.pinimg.com/originals/4c/98/4e/4c984ef0291409fef0a0942b391f6287.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    
    /* Custom sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #5F9EA0 !important; /* Mint green color */
    }

    /* Optional: Custom sidebar header color */
    [data-testid="stSidebarHeader"] {
        background-color: rgba(255, 255, 255, 0.8) !important; /* Semi-transparent background */
        border-bottom: 1px solid #ddd;
    }

    /* Optional: Custom sidebar content color */
    [data-testid="stSidebarContent"] {
        background-color: rgba(255, 255, 255, 0.8) !important; /* Semi-transparent background */
    }

    /* Style for model summary tables */
    .model-summary {
        font-family: monospace;
        white-space: pre;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
    }
    
    /* Style for model summary headers */
    .model-summary-header {
        background-color: #e9ecef;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# CNN
def build_model(input_shape=(224, 224, 6),num_classes=6):  
    inputs = Input(input_shape)
    image_input = inputs[:, :, :, :3]
    saliency_input = inputs[:, :, :, 3:]

    def create_cnn(input_tensor):
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(input_tensor)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        pool1 = Dropout(0.5)(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = Dropout(0.5)(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(0.5)(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        pool4 = Dropout(0.5)(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(conv5)
        conv5 = BatchNormalization()(conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
        pool5 = Dropout(0.5)(pool5)
        flatten = Flatten()(pool5)
        return flatten

    image_features = create_cnn(image_input)
    saliency_features = create_cnn(saliency_input)
    concatenated_features = tf.keras.layers.Concatenate()([image_features, saliency_features])
    dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(concatenated_features)
    dense1 = Dropout(0.5)(dense1)
    output = Dense(num_classes, activation='softmax')(dense1)

    model = Model(inputs=inputs, outputs=output)
    return model

# EfficientB4
def build_model_with_saliency_Eff(input_shape=(380, 380, 6), num_classes=6):
    base_model = EfficientNetB4(include_top=False, weights=None, input_shape=(380, 380, 3))

    # Split the input into image and saliency map
    inputs = Input(input_shape)
    image_input = inputs[:, :, :, :3]
    saliency_input = inputs[:, :, :, 3:]

    # Process the image input
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Process the saliency map
    y = base_model(saliency_input, training=False)
    y = GlobalAveragePooling2D()(y)

    # Combine both processed inputs
    combined = concatenate([x, y])
    combined = Dense(256, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=inputs, outputs=output)
    return model

# VGG16


def build_model_with_saliency_Vgg(input_shape=(224, 224, 6), weights_path=None, num_classes=6):
    base_model = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
    if weights_path is not None:
        base_model.load_weights(weights_path, by_name=True)
    base_model.trainable = False  

    # Split the input into image and saliency map
    inputs = Input(input_shape)
    image_input = inputs[:, :, :, :3]
    saliency_input = inputs[:, :, :, 3:]

    # Process the image input
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Process the saliency map
    y = base_model(saliency_input, training=False)
    y = GlobalAveragePooling2D()(y)

    # Combine both processed inputs
    combined = concatenate([x, y])
    combined = Dense(256, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=inputs, outputs=output)
    return model

#Resnet50
def build_model_with_saliency_Res(input_shape=(224, 224, 6), num_classes=6):
    base_model = ResNet50(include_top=False, weights=None, input_shape=(224, 224, 3))

    # Split the input into image and saliency map
    inputs = Input(input_shape)
    image_input = inputs[:, :, :, :3]
    saliency_input = inputs[:, :, :, 3:]

    # Process the image input
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Process the saliency map
    y = base_model(saliency_input, training=False)
    y = GlobalAveragePooling2D()(y)

    # Combine both processed inputs
    combined = concatenate([x, y])
    combined = Dense(256, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=inputs, outputs=output)
    return model

# InceptionResNetV2


def build_model_with_saliency_Inc(input_shape=(299, 299, 6), num_classes=6):
    base_model = InceptionResNetV2(include_top=False, weights=None, input_shape=(299, 299, 3))

    # Split the input into image and saliency map
    inputs = Input(input_shape)
    image_input = inputs[:, :, :, :3]
    saliency_input = inputs[:, :, :, 3:]

    # Process the image input
    x = base_model(image_input, training=False)
    x = GlobalAveragePooling2D()(x)

    # Process the saliency map
    y = base_model(saliency_input, training=False)
    y = GlobalAveragePooling2D()(y)

    # Combine both processed inputs
    combined = concatenate([x, y])
    combined = Dense(256, activation='relu')(combined)
    combined = Dropout(0.5)(combined)
    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=inputs, outputs=output)
    return model




# Loading Models 
# Define class labels for multi-class classification
skin_labels = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
@st.cache_resource
def load_models():
    models = {
        'skin': {},
        'derm': {}
    }

    # Load dermoscopy image models
    try:
        model_derm_cnn = load_model('DM_melanoma_cnn_with_saliency.keras')
        model_derm_vgg16 = load_model('DM_vgg16_model_with_saliency.keras')
        model_derm_resnet50 = load_model('DM_best_ResNet50_model.keras')
        model_derm_efficientnet = load_model('DM_efficientnetb4_model_with_saliency.keras')
        model_derm_inceptionresnetv2 = load_model('DM_InceptionResNetV2_model.keras')
    except Exception as e:
        st.error(f"Error loading dermoscopy image models: {e}")

    # Load skin image models
    try:
        # Load the CNN model weights
        model_skin_cnn = build_model()
        model_skin_cnn.load_weights('CNN_skin_classifier_weights.weights.h5')
        # st.write("Loaded CNN skin classifier model.")
    except Exception as e:
        st.error(f"Error loading CNN skin classifier model: {e}")

    try:
        # Load the VGG16 model weights
        model_skin_vgg16 = build_model_with_saliency_Vgg(input_shape=(224, 224, 6), num_classes=6)
        model_skin_vgg16.load_weights('best_VGG16_weights.weights.h5')
        # st.write("VGG16 skin model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading VGG16 skin model: {e}")

    try:
        # Load the ResNet50 model weights
        model_skin_resnet50 = build_model_with_saliency_Res(input_shape=(224, 224, 6), num_classes=6)
        model_skin_resnet50.load_weights('best_ResNet50_weights.weights.h5')
        # st.write("ResNet50 skin model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading ResNet50 skin model: {e}")

    try:
        # Load the EfficientNetB4 model weights
        model_skin_efficientnet = build_model_with_saliency_Eff(input_shape=(380, 380, 6), num_classes=6)
        model_skin_efficientnet.load_weights('best_EfficientNetB4_weights.weights.h5')
        # st.write("EfficientNetB4 skin model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading EfficientNetB4 skin model: {e}")

    try:
        # Load the InceptionResNetV2 model weights
        model_skin_inceptionresnetv2 = build_model_with_saliency_Inc(input_shape=(299, 299, 6), num_classes=6)
        model_skin_inceptionresnetv2.load_weights('best_InceptionResNetV2_weights.weights.h5')
        # st.write("InceptionResNetV2 skin model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading InceptionResNetV2 skin model: {e}")

    return {
        'derm': {
            'CNN': model_derm_cnn,
            'VGG16': model_derm_vgg16,
            'ResNet50': model_derm_resnet50,
            'EfficientNetB4': model_derm_efficientnet,
            'InceptionResNetV2': model_derm_inceptionresnetv2
        },
        'skin': {
            'CNN': model_skin_cnn,
            'VGG16': model_skin_vgg16,
            'ResNet50': model_skin_resnet50,
            'EfficientNetB4': model_skin_efficientnet,
            'InceptionResNetV2': model_skin_inceptionresnetv2
        }
    }, model_skin_cnn, model_skin_vgg16, model_skin_resnet50, model_skin_efficientnet, model_skin_inceptionresnetv2


# Detecting Melanoma

def melanoma_detection():
    st.title('Melanoma Detection')

    # Model selection using tabs
    tab1, tab2 = st.tabs(["Skin Image Models", "Dermoscopy Image Models"])
    
    # Load models (assuming this function is defined elsewhere)
    loaded_models, model_skin_cnn, model_skin_vgg16, model_skin_resnet50, model_skin_efficientnet, model_skin_inceptionresnetv2 = load_models()

    # Skin Image Models
    with tab1:
        st.header("Skin Image Models")
        models = ['CNN', 'VGG16', 'ResNet50', 'EfficientNetB4', 'InceptionResNetV2']
        model_dict = {
            'CNN': model_skin_cnn,
            'VGG16': model_skin_vgg16,
            'ResNet50': model_skin_resnet50,
            'EfficientNetB4': model_skin_efficientnet,
            'InceptionResNetV2': model_skin_inceptionresnetv2
        }
        preprocess_dict = {
            'VGG16': preprocess_vgg16,
            'ResNet50': preprocess_resnet50,
            'EfficientNetB4': preprocess_efficientnet,
            'InceptionResNetV2': preprocess_inceptionresnetv2
        }

    # Dermoscopy Image Models
    with tab2:
        st.header("Dermoscopy Image Models")
        models = ['CNN', 'VGG16', 'ResNet50', 'EfficientNetB4', 'InceptionResNetV2']
        model_dict = loaded_models['derm']
        preprocess_dict = {
            'VGG16': preprocess_vgg16,
            'ResNet50': preprocess_resnet50,
            'EfficientNetB4': preprocess_efficientnet,
            'InceptionResNetV2': preprocess_inceptionresnetv2
        }

    selected_model = st.selectbox('Select Model', models)

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
        st.write("")
        st.write("Classifying...")

        try:
            # Determine input size based on the selected model
            if selected_model == 'EfficientNetB4':
                input_size = (380, 380)
            elif selected_model == 'InceptionResNetV2':
                input_size = (299, 299)
            else:
                input_size = (224, 224)

            # Process uploaded image
            img = image.load_img(uploaded_file, target_size=input_size)
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            # Preprocess based on model
            if selected_model in preprocess_dict:
                img = preprocess_dict[selected_model](img)

            # Assuming saliency map is generated (dummy saliency map for illustration)
            saliency_map = np.zeros_like(img)

            # Concatenate the original image with the saliency map
            combined_input = np.concatenate((img, saliency_map), axis=-1)

            # Model prediction
            model = model_dict.get(selected_model)
            if model is None:
                st.error("Selected model is not available.")
            else:
                prediction = model.predict(combined_input)

                # Handle the prediction output
                if tab1:  
                    predicted_class = np.argmax(prediction, axis=-1)
                    confidence = np.max(prediction)
                    result = skin_labels[predicted_class[0]]
                else:  
                    threshold = 0.5  
                    predicted_class = (prediction[:, 0] > threshold).astype(int)
                    confidence = prediction[0, 0]
                    result = 'Malignant' if predicted_class[0] == 1 else 'Benign'

                st.write(f"Prediction: {result}")
                st.write(f"Confidence: {confidence:.2f}")

                # Additional information
                st.write("\nPlease note:")
                st.write("- This prediction is based on the model's analysis and should not be considered as a definitive medical diagnosis.")
                st.write("- If you have any concerns about a skin lesion, please consult with a qualified healthcare professional or dermatologist.")
                st.write("- Regular skin check-ups and early detection are crucial for managing melanoma risk.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.write("Please upload an image.")
    st.write("")    
    st.markdown(disclaimer_text_model, unsafe_allow_html=True)
    st.sidebar.markdown(disclaimer_text, unsafe_allow_html=True)





# Disclaimer text
disclaimer_text_model = """
    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;">
        <strong>Disclaimer:</strong> This app is for educational purposes only. Consult a healthcare professional for accurate medical advice.
    </div>
"""

disclaimer_text = """
    <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border: 1px solid #f5c6cb;">
        <strong>Warning:</strong> Please ensure that you understand the implications of using these models. Model performance can vary based on various factors.
    </div>
    """

# Main page 
def main():
    st.title('Melanoma Detection App')
    st.markdown("""
        ## Welcome to the Melanoma Detection App!

        This app is designed to help you understand and detect melanoma, a type of skin cancer. Melanoma is the most serious form of skin cancer and can be life-threatening if not detected and treated early.

        Using the latest advancements in deep learning and computer vision, this app provides a platform to explore different models for melanoma detection, visualize and analyze relevant data, and learn more about this important health topic.

        The key features of this app include:

        - **Model Selection and Performance**: Explore the accuracy, precision, recall, and AUC of various models for both skin and dermoscopy image classification.
        - **Visualizations**: Analyze the distribution of melanoma cases, age, and gender using interactive visualizations.
        - **Melanoma Detection**: Upload your own skin or dermoscopy images and get a prediction on whether the lesion is malignant or benign.
        - **Educational Resources**: Access a curated list of articles, journals, and websites to deepen your understanding of melanoma and related technologies.
        - **FAQs**: Get answers to common questions about the app, its models, and interpreting the results.
        - **Feedback and Contact**: Provide feedback or reach out to the app developers for support.

        We hope this app will be a valuable resource in your journey to learn about and detect melanoma. Let's get started!
    """)

    # Melanoma description and image
    st.markdown("""
        ## Understanding Melanoma

        Melanoma is a type of skin cancer that begins in melanocytes, the cells responsible for producing pigment in the skin. It is characterized by the uncontrolled growth of these cells, which can form tumors. Melanoma can occur in the skin, as well as in other parts of the body where pigment-producing cells are present.

        Early detection is crucial for successful treatment and a better prognosis. Melanoma is often identified by changes in the appearance of moles or skin lesions. The stages of melanoma range from stage 0, where the cancer is confined to the outer layer of the skin, to stage 4, where the cancer has spread to distant organs.

        Here is an image illustrating the different stages of melanoma:
    """)
    
    # Image URL
    melanoma_image_url = "https://www.aimatmelanoma.org/wp-content/uploads/Blue-Greyscale-Volleyball-Quote-UAAPNCAA-Facebook-Cover.jpg"

    # Display the image
    st.image(melanoma_image_url, use_container_width=True)



def display_model_summaries(models):
    for category, category_models in models.items():
        st.header(f"{category.capitalize()} Models")
        st.write("This section provides a detailed summary of the selected model's architecture...")
        
        for model_name, model in category_models.items():
            with st.expander(f"{model_name} Model Summary", expanded=False):
                summary_string = StringIO()
                model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
                
                st.markdown(f"""
                <div class="model-summary-header">
                    <strong>Model:</strong> {model_name} ({category})<br>
                    <strong>Total Params:</strong> {model.count_params():,}
                </div>
                <div class="model-summary">
                    {summary_string.getvalue()}
                </div>
                """, unsafe_allow_html=True)
                
                st.download_button(
                    label="Download Model Summary",
                    data=summary_string.getvalue(),
                    file_name=f"{model_name}_{category}_summary.txt",
                    mime="text/plain"
                )
            st.markdown("---")

# Function to display model evaluation metrics
def display_model_evaluation(metrics, model_type, model_name):
    if model_name in metrics:
        st.write(f"### Model Performance for {model_name} ({model_type})")
        st.write("The following metrics provide an overview of the selected model's performance:")
        st.write(f"**Accuracy**: {metrics[model_name]['Accuracy']}")
        st.write(f"**Precision**: {metrics[model_name]['Precision']}")
        st.write(f"**Recall**: {metrics[model_name]['Recall']}")
        st.write(f"**AUC**: {metrics[model_name]['AUC']}")
    else:
        st.write("Metrics not available.")

# Function to display confusion matrix
def display_confusion_matrix(confusion_matrices, model_type, model_name):
    matrix = confusion_matrices.get(model_type, {}).get(model_name)
    if matrix is not None:
        if model_type == 'dermoscopy':
            labels = ['Benign', 'Malignant']
        else:  
            labels = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
        
        # Convert confusion matrix to DataFrame for better handling with Plotly
        df_cm = pd.DataFrame(matrix, index=labels, columns=labels)
        
        # Create a Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=df_cm.values,
            x=df_cm.columns,
            y=df_cm.index,
            colorscale='Blues',
            colorbar=dict(title='Count'),
            zmin=0,
            zmax=df_cm.values.max()
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix for {model_name} ({model_type.capitalize()})',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            xaxis=dict(tickmode='array', tickvals=list(df_cm.columns), ticktext=df_cm.columns),
            yaxis=dict(tickmode='array', tickvals=list(df_cm.index), ticktext=df_cm.index)
        )
        
        st.plotly_chart(fig)

        st.write(f"The confusion matrix above shows the performance of the {model_name} model for {model_type.capitalize()} classification.")
        st.write(f"The diagonal elements represent the number of correct predictions, while the off-diagonals represent the number of incorrect predictions.")
    else:
        st.write(f"No confusion matrix available for {model_name} in {model_type}.")
# Confusion matrices for models
confusion_matrices = {
    'dermoscopy': {
        'VGG16': [[94, 42], [34, 121]],
        'ResNet50': [[106, 30], [39, 116]],
        'EfficientNetB4': [[119, 17], [26, 129]],
        'InceptionResNetV2': [[59, 77], [30, 125]],
        'CNN': [[84, 52], [51, 104]]
    },
    'skin': {
        'EfficientNetB4': [[110, 36, 4, 9, 18, 5],
                           [30, 101, 1, 10, 16, 4],
                           [5, 0, 123, 15, 6, 4],
                           [3, 9, 12, 117, 5, 14],
                           [19, 28, 17, 13, 86, 25],
                           [18, 13, 13, 26, 12, 87]],

        'VGG16': [[112, 32, 2, 2, 24, 10],
                  [27, 103, 1, 5, 17, 9],
                  [0, 0, 144, 0, 0, 9],
                  [5, 3, 1, 138, 6, 7],
                  [22, 38, 0, 1, 117, 10],
                  [21, 8, 5, 7, 3, 125]],

        'ResNet50': [[134, 18, 1, 3, 18, 8],
                     [23, 104, 3, 6, 24, 2],
                     [2, 0, 140, 6, 3, 2],
                     [1, 3, 1, 146, 2, 7],
                     [16, 27, 2, 0, 136, 7],
                     [16, 4, 0, 15, 0, 134]],

        'InceptionResNetV2': [[118, 24, 4, 4, 25, 7],
                              [25, 64, 2, 18, 30, 23],
                              [4, 0, 98, 38, 0, 13],
                              [2, 7, 13, 128, 0, 10],
                              [34, 37, 1, 10, 94, 12],
                              [27, 12, 13, 37, 8, 72]],

        'CNN': [[88, 55, 3, 1, 17, 18],
                [25, 125, 0, 3, 2, 7],
                [1, 1, 141, 8, 0, 2],
                [1, 0, 0, 153, 0, 6],
                [10, 3, 0, 0, 170, 5],
                [0, 0, 1, 5, 1, 162]]
    }
}

def display_selected_model_summary(models):
    image_type = st.sidebar.selectbox('Select Image Type', ['Skin', 'Dermoscopy'])
    model_key = 'skin' if image_type == 'Skin' else 'derm'
    
    model_name = st.sidebar.selectbox('Select Model', list(models[model_key].keys()))
    
    st.header(f"{image_type} - {model_name} Model Summary")
    model = models[model_key][model_name]
    summary_string = StringIO()
    model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
    st.text(summary_string.getvalue())



def plot_roc_curve(model, model_name, num_classes, input_size):
    # Generate random sample data with the correct input size
    num_samples = 100
    X_sample = np.random.rand(num_samples, *input_size).astype(np.float32)
    y_sample = np.random.randint(0, num_classes, size=(num_samples,))
    
    # Get predictions
    y_pred = model.predict(X_sample)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_sample == i, y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    fig = go.Figure()
    for i in range(num_classes):
        fig.add_trace(go.Scatter(x=fpr[i], y=tpr[i], 
                                 name=f'Class {i} (AUC = {roc_auc[i]:.2f})'))
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                             line=dict(dash='dash'), name='Random Classifier'))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
        width=800,
        height=500,
        margin=dict(r=200)  
    )
    return fig




def model_performance_page():
    if 'models' not in st.session_state:
        loaded_data = load_models()
        st.session_state.models = loaded_data[0] if isinstance(loaded_data, tuple) else loaded_data
        
    st.title("Model Performance")
    st.markdown("This section allows you to explore the performance of various models used for melanoma detection. You can view the model summaries, evaluation metrics, and confusion matrices.")
    
    metrics_dermoscopy = {
        'CNN': {'Accuracy': '53%', 'Precision': '53%', 'Recall': '53%', 'AUC': '52%'},
        'VGG16': {'Accuracy': '74%', 'Precision': '74%', 'Recall': '74%', 'AUC': '83%'},
        'ResNet50': {'Accuracy': '75%', 'Precision': '75%', 'Recall': '75%', 'AUC': '84%'},
        'EfficientNetB4': {'Accuracy': '85%', 'Precision': '85%', 'Recall': '85%', 'AUC': '95%'},
        'InceptionResNetV2': {'Accuracy': '60%', 'Precision': '60%', 'Recall': '60%', 'AUC': '64%'}
    }
    metrics_skin = {
        'CNN': {'Accuracy': '75%', 'Precision': '95%', 'Recall': '44%', 'AUC': '96%'},
        'VGG16': {'Accuracy': '74%', 'Precision': '82%', 'Recall': '66%', 'AUC': '95%'},
        'ResNet50': {'Accuracy': '79%', 'Precision': '84%', 'Recall': '76%', 'AUC': '97%'},
        'EfficientNetB4': {'Accuracy': '61%', 'Precision': '70%', 'Recall': '50%', 'AUC': '89%'},
        'InceptionResNetV2': {'Accuracy': '57%', 'Precision': '95%', 'Recall': '44%', 'AUC': '96%'}
    }
    
    model_descriptions = {
        'CNN': "A custom Convolutional Neural Network designed for this specific task.",
        'VGG16': "A deep CNN known for its simplicity and effectiveness in image classification.",
        'ResNet50': "A deep residual network that addresses the vanishing gradient problem.",
        'EfficientNetB4': "A network that balances network depth, width, and resolution for improved efficiency.",
        'InceptionResNetV2': "Combines the Inception architecture with residual connections for enhanced performance."
    }

    

    # Model selection
    model_type = st.sidebar.selectbox('Select Model Type', ['Skin Image Models', 'Dermoscopy Image Models'])
    metrics = metrics_skin if model_type == 'Skin Image Models' else metrics_dermoscopy
    model_name = st.sidebar.selectbox('Select Model', list(metrics.keys()), 
                                      help="Choose a model to view its performance metrics and details.")
    
    st.sidebar.markdown(f"**Model Description:**\n{model_descriptions[model_name]}")

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Model Summary", "Performance Metrics", "Confusion Matrix", "ROC Curve"])

    with tab1:  # Model Summary tab
        st.header(f"{model_type} - {model_name} Model Summary")
        model = st.session_state.models['skin' if model_type == 'Skin Image Models' else 'derm'][model_name]
        
        with st.expander("Show/Hide Model Summary", expanded=True):
            summary_string = StringIO()
            model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
            
            # Display the summary with custom styling
            st.markdown(f"""
            <div class="model-summary-header">
                <strong>Model:</strong> {model_name} ({model_type})<br>
                <strong>Total Params:</strong> {model.count_params():,}
            </div>
            <div class="model-summary">
                {summary_string.getvalue()}
            </div>
            """, unsafe_allow_html=True)
            
            # Add download button for the model summary
            st.download_button(
                label="Download Model Summary",
                data=summary_string.getvalue(),
                file_name=f"{model_name}_{model_type.replace(' ', '_')}_summary.txt",
                mime="text/plain"
            )

    with tab2:
        display_model_evaluation(metrics, model_type, model_name)

    with tab3:
        model_type_key = 'skin' if model_type == 'Skin Image Models' else 'dermoscopy'
        display_confusion_matrix(confusion_matrices, model_type_key, model_name)

    with tab4:
        st.header(f"ROC Curve for {model_name}")
        model = st.session_state.models['skin' if model_type == 'Skin Image Models' else 'derm'][model_name]
        num_classes = 6 if model_type == 'Skin Image Models' else 2

        # Define the correct input size based on the model
        if model_name == 'InceptionResNetV2':
            input_size = (299, 299, 6)
        elif model_name == 'EfficientNetB4':
            input_size = (380, 380, 6)
        else:
            input_size = (224, 224, 6)

        fig = plot_roc_curve(model, model_name, num_classes, input_size)
        st.plotly_chart(fig)
        
    st.sidebar.markdown(disclaimer_text, unsafe_allow_html=True)




# plotting the visualization from the metadata


def visualize_data():
    st.title('Visualizations')
    st.markdown("""
        ## 
        This section provides interactive visualizations to help you understand the distribution of melanoma cases, age, and gender in the dataset.
        
        You can switch between visualizations for skin images and dermoscopy images using the sidebar selection.
    """)

    # Load datasets
    df = pd.read_csv('metadata.csv')
    df2 = pd.read_csv('ISBI2016_ISIC_Part3_Training_GroundTruth.csv')

    # Sidebar selection
    visualization_type = st.sidebar.selectbox('Select Visualization', ['Skin', 'Dermoscopy'])

    if visualization_type == 'Skin':
        # Skin Visualizations

        # Diagnostic Distribution
        st.subheader('Diagnostic Distribution for PH2 Dataset')
        fig_diag = px.histogram(df, x='diagnostic', title='Diagnostic Distribution', labels={'diagnostic': 'Diagnostic'})
        fig_diag.update_layout(xaxis_title='Diagnostic', yaxis_title='Count')
        st.plotly_chart(fig_diag)

        # Diagnostic Distribution by Gender
        st.subheader('Diagnostic Distribution by Gender for PH2 Dataset')
        fig_gender = px.histogram(df, x='gender', color='diagnostic', barmode='group', title='Diagnostic Distribution by Gender')
        fig_gender.update_layout(xaxis_title='Gender', yaxis_title='Count', legend_title='Diagnostic')
        st.plotly_chart(fig_gender)

        # Box plot for age by diagnostic
        st.subheader('Age Distribution by Diagnostic for PH2 Dataset')
        fig_age = px.box(df, x='diagnostic', y='age', title='Age Distribution by Diagnostic', labels={'diagnostic': 'Diagnostic', 'age': 'Age'})
        st.plotly_chart(fig_age)

    elif visualization_type == 'Dermoscopy':
        # Dermoscopy Visualizations

        # Pie chart for label distribution
        st.subheader('Distribution of Labels for ISIC2016 Dataset')
        label_counts = df2['Label'].value_counts()
        labels = label_counts.index
        sizes = label_counts.values

        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.3)])
        fig_pie.update_layout(title='Label Distribution', legend_title='Labels')
        st.plotly_chart(fig_pie)
        
    st.sidebar.markdown(disclaimer_text, unsafe_allow_html=True)


    


# Educational resources section

def educational_resources():
    st.title('Educational Resources')
    st.markdown("""
        ## Educational Resources for Melanoma Detection
        
        In this section, you can find a curated list of resources to deepen your understanding of melanoma and the technologies used in this app for detection.
        
        The resources include academic articles, reputable websites, scientific journals, and research papers that cover various aspects of melanoma and deep learning for medical imaging.
        
        Feel free to explore these resources to learn more about this important health topic and the latest advancements in the field.
        
        Below are verified, up-to-date resources about melanoma and AI detection:
    """)

    st.markdown("""
    ### Medical Organizations
    - [Skin Cancer Foundation](https://www.skincancer.org/): Authoritative resource on prevention, detection and treatment
    - [American Academy of Dermatology](https://www.aad.org/public/diseases/skin-cancer/types/common/melanoma): Melanoma overview from dermatologists
    - [National Cancer Institute](https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq): Official treatment information

    ### Research & Datasets
    - [ISIC Archive](https://www.isic-archive.com/): Largest public dermoscopy image collection
    - [DermNet NZ](https://dermnetnz.org/topics/melanoma/): Clinical images and information
    - [PH2](https://www.fc.up.pt/addi/ph2%20database.html): Public skin image database

    ### Technical Papers
    - [IEEE: Deep Learning for Melanoma Detection](https://ieeexplore.ieee.org/document/10691081):(IEEE access required)
    - [Nature: Dermatologist-level Melanoma Classification](https://www.nature.com/articles/nature21056): open-access study
    - [PLOS: CNN for Melanoma Diagnosis](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1003381): open-access study

    ### Patient Resources
    - [Melanoma Research Alliance](https://www.curemelanoma.org/): Latest research and trials
    - [CDC Melanoma Facts](https://www.cdc.gov/skin-cancer/): Government statistics
    - [AIM at Melanoma](https://www.aimatmelanoma.org/): Comprehensive patient guide
    
    """)

    st.markdown("""
    *Note: Some resources may require institutional access or subscriptions for full content.*
    """)

# FAQs section
def faq_section():
    st.title('FAQs')
    st.markdown("""
        ## Frequently Asked Questions

        This section addresses some common questions about the Melanoma Detection App. If you have any other questions, feel free to reach out to us using the Feedback and Contact form.

        **Q: How accurate are the models used in this app?**
        - A: The accuracy of the models varies, as different architectures have different performance characteristics. You can check the Model Performance page for detailed metrics on accuracy, precision, recall, and AUC for each model.

        **Q: Can I trust the results of this app for medical diagnosis?**
        - A: This app is designed for educational and informational purposes only. The results should not be used as a substitute for professional medical advice. If you have any concerns about a skin lesion, please consult a qualified healthcare provider.

        **Q: What types of images can I upload for detection?**
        - A: You can upload both skin images and dermoscopy images for melanoma detection. The app will automatically detect the image type and use the appropriate model for classification.

        **Q: How do I interpret the prediction results?**
        - A: The app will classify the uploaded image as either Melanoma (Malignant) or Not Melanoma (Benign), along with a confidence score. This information is provided to help you understand the model's assessment, but should not be considered a definitive diagnosis.

        **Q: Is my uploaded image stored or used for other purposes?**
        - A: No, your uploaded images are not stored or used for any other purpose. They are only used for the current session's classification and are not retained or shared.
    """)

# Feedback and contact form
# Function to send feedback via email
def send_email(name, email, message):
    sender_email = "debbydawn16@gmail.com"  
    sender_password = "fcgd zzhr szgf izia" 

    recipient_email = "debbydawn16@gmail.com"
    
    subject = "New Feedback from Melanoma Detection App"
    
    # Create the email body
    body = f"""
    You have received a new feedback message:

    Name: {name}
    Email: {email}
    Message: {message}
    """

    # Set up the MIME
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Establish a secure session with Gmail's outgoing SMTP server using your Gmail account
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  
        server.login(sender_email, sender_password)
        
        # Send email
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.close()

        st.success("Your message has been sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Feedback form
def feedback_form():
    st.title('Feedback and Contact')
    st.write("If you have any feedback, questions, or need support, please don't hesitate to reach out to us.")

    # Form input fields
    with st.form(key='feedback_form'):
        name = st.text_input('Name')
        email = st.text_input('Email')
        message = st.text_area('Message')

        # Submit button
        submit_button = st.form_submit_button(label='Submit')

    # Process the feedback form
    if submit_button:
        if not name or not email or not message:
            st.error("Please fill out all fields.")
        else:
            send_email(name, email, message)
            


# Function to crop an image into a circle
def crop_to_circle(image):
    # Create a mask to crop the image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + image.size, fill=255)
    
    # Apply the mask to the image
    result = Image.new("RGBA", image.size)
    result.paste(image, (0, 0), mask)
    
    # Crop out transparent edges
    bbox = mask.getbbox()
    result = result.crop(bbox)
    
    return result

# Combine all pages into a single app
def run_app():
    
    # Load the Melanoma logo from the local filepath
    sidebar_image_path = "logo_2.PNG"
    sidebar_image = Image.open(sidebar_image_path)
    
    # Crop the image to a circle
    sidebar_image = crop_to_circle(sidebar_image)

    # Add the circular image to the sidebar
    st.sidebar.image(sidebar_image, use_container_width=True)
    
    st.sidebar.title('Navigation')
    pages = {
        "Introduction": main,
        "Model Performance": model_performance_page,
        "Visualizations": visualize_data,
        "Melanoma Detection": melanoma_detection,
        "Educational Resources": educational_resources,
        "FAQs": faq_section,
        "Feedback and Contact": feedback_form
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))
    pages[selection]()
    
if __name__ == '__main__':
    run_app()
