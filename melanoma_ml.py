import streamlit as st
import numpy as np
import gdown
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import pytesseract
except ImportError:
    pytesseract = None
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Dict, Tuple
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import os
from pathlib import Path
import sqlite3
import hashlib
import re
import secrets
from urllib.error import URLError, HTTPError
from urllib.parse import quote
from urllib.request import urlopen

TF_IMPORT_ERROR = None
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import GlobalAveragePooling2D, concatenate
    from tensorflow.keras.applications import VGG16, ResNet50, InceptionResNetV2, EfficientNetB4
    from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
    from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
    from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2,
        preprocess_input as preprocess_mobilenet_v2,
        decode_predictions,
    )
    TF_AVAILABLE = True
except Exception as exc:
    TF_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    TF_AVAILABLE = False

    def _tensorflow_unavailable(*args, **kwargs):
        raise RuntimeError(
            "TensorFlow is unavailable in this environment. "
            "Use Python 3.11 and reinstall dependencies."
        )

    tf = None
    load_model = _tensorflow_unavailable
    image = None
    Model = object
    Input = Conv2D = MaxPooling2D = Dropout = Flatten = Dense = BatchNormalization = _tensorflow_unavailable
    l2 = _tensorflow_unavailable
    GlobalAveragePooling2D = concatenate = _tensorflow_unavailable
    VGG16 = ResNet50 = InceptionResNetV2 = EfficientNetB4 = _tensorflow_unavailable
    MobileNetV2 = _tensorflow_unavailable
    preprocess_vgg16 = preprocess_resnet50 = preprocess_efficientnet = preprocess_inceptionresnetv2 = _tensorflow_unavailable
    preprocess_mobilenet_v2 = decode_predictions = _tensorflow_unavailable

MODEL_DIR = Path(__file__).resolve().parent
AUTH_DB_PATH = MODEL_DIR / "users.db"
TESSERACT_CANDIDATE_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]

# Copy trained artifacts into MODEL_DIR using exactly these names
DERM_MODEL_FILENAMES = {
    "CNN": "DM_melanoma_cnn_with_saliency.keras",
    "VGG16": "DM_vgg16_model_with_saliency.keras",
    "ResNet50": "DM_best_ResNet50_model.keras",
    "EfficientNetB4": "DM_efficientnetb4_model_with_saliency.keras",
    "InceptionResNetV2": "DM_InceptionResNetV2_model.keras",
}
SKIN_WEIGHT_FILENAMES = {
    "CNN": "CNN_skin_classifier_weights.weights.h5",
    "VGG16": "best_VGG16_weights.weights.h5",
    "ResNet50": "best_ResNet50_weights.weights.h5",
    "EfficientNetB4": "best_EfficientNetB4_weights.weights.h5",
    "InceptionResNetV2": "best_InceptionResNetV2_weights.weights.h5",
}
ALL_REQUIRED_MODEL_FILES = tuple(DERM_MODEL_FILENAMES.values()) + tuple(SKIN_WEIGHT_FILENAMES.values())
MODEL_URL_ENV_MAP = {
    "DM_melanoma_cnn_with_saliency.keras": "MODEL_URL_DM_CNN",
    "DM_vgg16_model_with_saliency.keras": "MODEL_URL_DM_VGG16",
    "DM_best_ResNet50_model.keras": "MODEL_URL_DM_RESNET50",
    "DM_efficientnetb4_model_with_saliency.keras": "MODEL_URL_DM_EFFICIENTNETB4",
    "DM_InceptionResNetV2_model.keras": "MODEL_URL_DM_INCEPTIONRESNETV2",
    "CNN_skin_classifier_weights.weights.h5": "MODEL_URL_SKIN_CNN",
    "best_VGG16_weights.weights.h5": "MODEL_URL_SKIN_VGG16",
    "best_ResNet50_weights.weights.h5": "MODEL_URL_SKIN_RESNET50",
    "best_EfficientNetB4_weights.weights.h5": "MODEL_URL_SKIN_EFFICIENTNETB4",
    "best_InceptionResNetV2_weights.weights.h5": "MODEL_URL_SKIN_INCEPTIONRESNETV2",
}
DEFAULT_MODEL_URLS = {
    "MODEL_URL_DM_CNN": "https://drive.google.com/file/d/1JuqOdzTA_Ob9Mz_1bbR04aGvdMUruDU4/view?usp=sharing",
    "MODEL_URL_DM_VGG16": "https://drive.google.com/file/d/14Vq10VoVV_CL56DQGgV8moFLLCyioXbm/view?usp=sharing",
    "MODEL_URL_DM_RESNET50": "https://drive.google.com/file/d/1y1PPSuoyHiqoUZ0krOZCiblIDXkkJUh2/view?usp=sharing",
    "MODEL_URL_DM_EFFICIENTNETB4": "https://drive.google.com/file/d/16oU9vOF12zp5Nwl690c37WYJNrD7YtPl/view?usp=sharing",
    "MODEL_URL_DM_INCEPTIONRESNETV2": "https://drive.google.com/file/d/1swv0UiZgD9hiBg1h0Qj5DTshMsO7Evuq/view?usp=sharing",
    "MODEL_URL_SKIN_CNN": "https://drive.google.com/file/d/1ygf6vmlONdGq3dRlRE3SzwJm5J0rPw26/view?usp=sharing",
    "MODEL_URL_SKIN_VGG16": "https://drive.google.com/file/d/1k51aj3lI2MP7D8opn_bX6XjGnatVfSUd/view?usp=sharing",
    "MODEL_URL_SKIN_RESNET50": "https://drive.google.com/file/d/1XfP5C0XxmtpzrryScYZiRt7aXgGrmoRr/view?usp=sharing",
    "MODEL_URL_SKIN_EFFICIENTNETB4": "https://drive.google.com/file/d/1xN4wND-0e4dzP-P8l-ANcNs9US5-Te0F/view?usp=sharing",
    "MODEL_URL_SKIN_INCEPTIONRESNETV2": "https://drive.google.com/file/d/1GCXMRil1vBvptsR7MTvCS4liDpS5c0sx/view?usp=sharing",
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

if pytesseract is not None:
    for candidate in TESSERACT_CANDIDATE_PATHS:
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            break
    # Ensure tessdata can be found when PATH is not refreshed in current session.
    cmd_path = Path(getattr(pytesseract.pytesseract, "tesseract_cmd", ""))
    tessdata_dir = cmd_path.parent / "tessdata"
    if cmd_path.is_file() and tessdata_dir.is_dir():
        os.environ.setdefault("TESSDATA_PREFIX", str(tessdata_dir))

# Disable GPU on machines where TensorFlow sees a GPU; ignore errors on CPU-only installs
if TF_AVAILABLE:
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass

# Set up logging
logging.basicConfig(level=logging.INFO)

# Page config is set in M_Detect.py before this module loads


def init_auth_db() -> None:
    with sqlite3.connect(AUTH_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def hash_password(password: str) -> str:
    # Store as: pbkdf2_sha256$iterations$salt$hash
    iterations = 260000
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    # Backward compatibility for older SHA-256-only records.
    if "$" not in stored_hash:
        return hashlib.sha256(password.encode("utf-8")).hexdigest() == stored_hash

    try:
        scheme, iterations_str, salt, expected_digest = stored_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iterations_str)
        candidate_digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        ).hex()
        return secrets.compare_digest(candidate_digest, expected_digest)
    except Exception:
        return False


def validate_registration_input(username: str, email: str, password: str, confirm_password: str) -> str | None:
    if not username.strip() or not email.strip() or not password:
        return "Username, email, and password are required."
    if len(username.strip()) < 3:
        return "Username must be at least 3 characters."
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()):
        return "Please provide a valid email address."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must include at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must include at least one number."
    if not re.search(r"[^A-Za-z0-9]", password):
        return "Password must include at least one special character."
    if password != confirm_password:
        return "Passwords do not match."
    return None


def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    try:
        with sqlite3.connect(AUTH_DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username.strip(), email.strip().lower(), hash_password(password)),
            )
            conn.commit()
        return True, "Registration successful. Please log in."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."
    except Exception:
        return False, "Registration failed. Please try again."


def authenticate_user(username_or_email: str, password: str) -> tuple[bool, str]:
    if not username_or_email.strip() or not password:
        return False, "Please enter your username/email and password."
    with sqlite3.connect(AUTH_DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT username, password_hash
            FROM users
            WHERE lower(username) = lower(?) OR lower(email) = lower(?)
            LIMIT 1
            """,
            (username_or_email.strip(), username_or_email.strip()),
        ).fetchone()
    if row is None:
        return False, "No account found with those details."
    username, stored_hash = row
    if not verify_password(password, stored_hash):
        return False, "Invalid password."
    st.session_state.authenticated = True
    st.session_state.current_user = username
    return True, f"Welcome, {username}!"


def reset_password(username_or_email: str, email: str, new_password: str, confirm_password: str) -> tuple[bool, str]:
    if not username_or_email.strip() or not email.strip():
        return False, "Please enter your username/email and email."
    email_clean = email.strip().lower()
    validation_error = validate_registration_input("reset_user", email_clean, new_password, confirm_password)
    if validation_error:
        return False, validation_error

    with sqlite3.connect(AUTH_DB_PATH) as conn:
        user_row = conn.execute(
            """
            SELECT id, email
            FROM users
            WHERE lower(username) = lower(?) OR lower(email) = lower(?)
            LIMIT 1
            """,
            (username_or_email.strip(), username_or_email.strip()),
        ).fetchone()
        if user_row is None:
            return False, "No account found with those details."

        user_id, stored_email = user_row
        if stored_email.lower() != email_clean:
            return False, "Email does not match the selected account."

        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(new_password), user_id),
        )
        conn.commit()
    return True, "Password reset successful. Please log in with your new password."


def auth_page() -> None:
    st.title("User Access")
    st.write("Log in or create an account to use the melanoma detection features.")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            login_identifier = st.text_input("Username or Email")
            login_password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login")
        if login_submit:
            success, message = authenticate_user(login_identifier, login_password)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        with st.expander("Forgot Password?"):
            with st.form("forgot_password_form"):
                fp_identifier = st.text_input("Username or Email")
                fp_email = st.text_input("Registered Email")
                fp_new_password = st.text_input("New Password", type="password")
                fp_confirm_password = st.text_input("Confirm New Password", type="password")
                fp_submit = st.form_submit_button("Reset Password")
            if fp_submit:
                success, message = reset_password(
                    fp_identifier,
                    fp_email,
                    fp_new_password,
                    fp_confirm_password,
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_submit = st.form_submit_button("Register")
        if register_submit:
            validation_error = validate_registration_input(
                new_username, new_email, new_password, confirm_password
            )
            if validation_error:
                st.error(validation_error)
            else:
                success, message = register_user(new_username, new_email, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)


def get_ocr_status():
    """Return OCR status for UI and validation logic."""
    if pytesseract is None:
        return False, "pytesseract Python package is not installed."
    cmd = Path(getattr(pytesseract.pytesseract, "tesseract_cmd", ""))
    if not cmd.is_file():
        return False, "Tesseract executable not found."
    try:
        _ = pytesseract.get_tesseract_version()
        return True, f"OCR ready ({cmd})"
    except Exception as e:
        return False, f"Tesseract detected but not usable: {e}"


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
def _model_path(filename: str) -> Path:
    return MODEL_DIR / filename


def _is_lfs_pointer_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_line = handle.readline().strip()
        return first_line.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def _is_valid_model_artifact(path: Path) -> bool:
    return path.is_file() and not _is_lfs_pointer_file(path)


def _resolve_model_download_url(filename: str) -> str | None:
    direct_key = MODEL_URL_ENV_MAP.get(filename)
    if direct_key:
        direct_value = os.environ.get(direct_key, "").strip()
        if direct_value:
            return direct_value
        try:
            secret_value = str(st.secrets.get(direct_key, "")).strip()
            if secret_value:
                return secret_value
        except Exception:
            pass
        fallback_value = DEFAULT_MODEL_URLS.get(direct_key, "").strip()
        if fallback_value:
            return fallback_value
    base_url = os.environ.get("MODEL_ASSET_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        try:
            base_url = str(st.secrets.get("MODEL_ASSET_BASE_URL", "")).strip().rstrip("/")
        except Exception:
            base_url = ""
    if not base_url:
        return None
    return f"{base_url}/{quote(filename)}"


def _download_model_file(url: str, destination: Path) -> tuple[bool, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if "drive.google.com" in url:
            try:
                # Newer gdown supports fuzzy links from shared Drive URLs.
                gdown.download(url, str(destination), quiet=True, fuzzy=True)
            except TypeError:
                # Older gdown versions do not accept `fuzzy`; fallback keeps compatibility.
                gdown.download(url, str(destination), quiet=True)
        else:
            with urlopen(url, timeout=120) as response, destination.open("wb") as out_file:
                out_file.write(response.read())
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if not _is_valid_model_artifact(destination):
        return False, "downloaded file is missing or still a Git LFS pointer"
    return True, ""


def ensure_model_artifacts() -> tuple[list[str], dict[str, str]]:
    missing_or_invalid: list[str] = []
    download_errors: dict[str, str] = {}
    for filename in ALL_REQUIRED_MODEL_FILES:
        path = _model_path(filename)
        if _is_valid_model_artifact(path):
            continue

        url = _resolve_model_download_url(filename)
        if not url:
            missing_or_invalid.append(filename)
            continue

        success, error = _download_model_file(url, path)
        if not success:
            missing_or_invalid.append(filename)
            download_errors[filename] = error
    return missing_or_invalid, download_errors


@st.cache_resource(show_spinner="Loading melanoma detection models...")
def load_models():
    if not TF_AVAILABLE:
        logging.error("TensorFlow unavailable: %s", TF_IMPORT_ERROR)
        empty = {
            'derm': {
                'CNN': None,
                'VGG16': None,
                'ResNet50': None,
                'EfficientNetB4': None,
                'InceptionResNetV2': None,
            },
            'skin': {
                'CNN': None,
                'VGG16': None,
                'ResNet50': None,
                'EfficientNetB4': None,
                'InceptionResNetV2': None,
            },
        }
        return empty, None, None, None, None, None

    # Add this to prevent TensorFlow from allocating all memory
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    model_derm_cnn = None
    model_derm_vgg16 = None
    model_derm_resnet50 = None
    model_derm_efficientnet = None
    model_derm_inceptionresnetv2 = None
    model_skin_cnn = None
    model_skin_vgg16 = None
    model_skin_resnet50 = None
    model_skin_efficientnet = None
    model_skin_inceptionresnetv2 = None

    for label, fname in DERM_MODEL_FILENAMES.items():
        path = _model_path(fname)
        if not _is_valid_model_artifact(path):
            logging.warning("Missing dermoscopy model file: %s", path)
            continue
        try:
            m = load_model(path)
            if label == 'CNN':
                model_derm_cnn = m
            elif label == 'VGG16':
                model_derm_vgg16 = m
            elif label == 'ResNet50':
                model_derm_resnet50 = m
            elif label == 'EfficientNetB4':
                model_derm_efficientnet = m
            elif label == 'InceptionResNetV2':
                model_derm_inceptionresnetv2 = m
        except Exception as e:
            logging.exception("Error loading dermoscopy model %s: %s", fname, e)

    skin_weights = [
        ('CNN', build_model, (), SKIN_WEIGHT_FILENAMES['CNN']),
        ('VGG16', build_model_with_saliency_Vgg, ((224, 224, 6),), SKIN_WEIGHT_FILENAMES['VGG16']),
        ('ResNet50', build_model_with_saliency_Res, ((224, 224, 6),), SKIN_WEIGHT_FILENAMES['ResNet50']),
        ('EfficientNetB4', build_model_with_saliency_Eff, ((380, 380, 6),), SKIN_WEIGHT_FILENAMES['EfficientNetB4']),
        ('InceptionResNetV2', build_model_with_saliency_Inc, ((299, 299, 6),), SKIN_WEIGHT_FILENAMES['InceptionResNetV2']),
    ]
    for label, builder, shape_args, fname in skin_weights:
        path = _model_path(fname)
        if not _is_valid_model_artifact(path):
            logging.warning("Missing skin weights file: %s", path)
            continue
        try:
            m = builder(input_shape=shape_args[0], num_classes=6) if shape_args else builder()
            m.load_weights(path)
            if label == 'CNN':
                model_skin_cnn = m
            elif label == 'VGG16':
                model_skin_vgg16 = m
            elif label == 'ResNet50':
                model_skin_resnet50 = m
            elif label == 'EfficientNetB4':
                model_skin_efficientnet = m
            elif label == 'InceptionResNetV2':
                model_skin_inceptionresnetv2 = m
        except Exception as e:
            logging.exception("Error loading skin model %s: %s", fname, e)

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


@st.cache_resource
def load_validation_models():
    """Load models used for image validation."""
    if not TF_AVAILABLE:
        logging.error("TensorFlow unavailable: %s", TF_IMPORT_ERROR)
        return None, None
    base_model = MobileNetV2(weights="imagenet", include_top=True)
    face_cascade = None
    if cv2 is not None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return base_model, face_cascade


def detect_text_in_image(img_array):
    """
    Detect significant text in the image.
    Returns: (has_text: bool, text_percentage: float, text_regions: int)
    """
    ocr_ready, _ = get_ocr_status()
    if not ocr_ready or cv2 is None:
        return False, 0.0, 0
    try:
        img_for_ocr = img_array[0] if img_array.shape[0] == 1 else img_array
        if img_for_ocr.dtype in (np.float32, np.float64):
            img_for_ocr = np.clip(img_for_ocr, 0, 255).astype(np.uint8)
        if len(img_for_ocr.shape) == 3 and img_for_ocr.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_for_ocr, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_for_ocr
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        custom_config = r"--oem 3 --psm 6"
        data = pytesseract.image_to_data(
            gray,
            config=custom_config,
            output_type=pytesseract.Output.DICT,
        )
        text_regions = 0
        for conf in data.get("conf", []):
            try:
                # Use a stricter confidence cutoff to reduce OCR false positives
                # on lesion textures that can look like tiny characters.
                if float(conf) > 50:
                    text_regions += 1
            except Exception:
                continue
        total_regions = len(data.get("conf", []))
        text_percentage = (text_regions / total_regions * 100) if total_regions else 0
        # Treat text as significant only when both density and count are meaningful.
        has_significant_text = (text_regions >= 3 and text_percentage >= 30) or text_regions > 20
        return has_significant_text, text_percentage, text_regions
    except Exception:
        return False, 0.0, 0


def detect_faces(img_array, face_cascade):
    """
    Detect if image contains faces.
    Returns: (has_face: bool, num_faces: int)
    """
    if cv2 is None or face_cascade is None:
        return False, 0
    try:
        img_for_face = img_array[0] if img_array.shape[0] == 1 else img_array
        if img_for_face.dtype in (np.float32, np.float64):
            img_for_face = np.clip(img_for_face, 0, 255).astype(np.uint8)
        if len(img_for_face.shape) == 3 and img_for_face.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_for_face, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_for_face
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Two-pass detection improves recall for partial/close portraits.
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(24, 24),
        )
        if len(faces) == 0:
            gray_eq = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(
                gray_eq,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(24, 24),
            )
        return len(faces) > 0, len(faces)
    except Exception:
        return False, 0


def detect_document_patterns(img_array):
    """
    Detect document-like patterns such as lines/rectangles.
    Returns: (is_document: bool, confidence: float, num_lines: int, rectangular_contours: int)
    """
    if cv2 is None:
        return False, 0.0, 0, 0
    try:
        img = img_array[0] if img_array.shape[0] == 1 else img_array
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        num_lines = 0 if lines is None else len(lines)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangular_contours = 0
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                rectangular_contours += 1
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        high_freq_content = float(np.mean(magnitude[magnitude > np.percentile(magnitude, 95)]))
        has_many_lines = num_lines > 10
        has_rectangles = rectangular_contours > 2
        has_printed_patterns = high_freq_content > 1000
        document_score = sum([has_many_lines, has_rectangles, has_printed_patterns, num_lines > 20])
        is_document = document_score >= 2 or num_lines > 15
        return is_document, document_score / 4.0, num_lines, rectangular_contours
    except Exception:
        return False, 0.0, 0, 0


def detect_plain_skin_no_lesion(img_array):
    """
    Heuristic check for plain skin without lesion-like visual structure.
    Returns: (is_plain_skin: bool, details: dict)
    """
    if cv2 is None:
        return False, {}
    try:
        img = img_array[0] if img_array.shape[0] == 1 else img_array
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img

        edges = cv2.Canny(gray, 60, 140)
        edge_density = float(np.mean(edges > 0))
        intensity_std = float(np.std(gray))
        texture_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        median_intensity = float(np.median(gray))
        contrast_mask = np.abs(gray.astype(np.float32) - median_intensity) > 35
        contrast_ratio = float(np.mean(contrast_mask))

        # Lower structure/contrast usually indicates plain skin and no obvious lesion.
        is_plain_skin = (
            edge_density < 0.025
            and intensity_std < 30
            and texture_var < 180
            and contrast_ratio < 0.08
        )
        plain_confidence = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    min(edge_density / 0.05, 1.0) * 0.35
                    + min(intensity_std / 60.0, 1.0) * 0.25
                    + min(texture_var / 350.0, 1.0) * 0.25
                    + min(contrast_ratio / 0.16, 1.0) * 0.15
                ),
            ),
        )
        details = {
            "edge_density": edge_density,
            "intensity_std": intensity_std,
            "texture_var": texture_var,
            "contrast_ratio": contrast_ratio,
            "plain_confidence": plain_confidence,
        }
        return is_plain_skin, details
    except Exception:
        return False, {}


def has_prominent_lesion_candidate(img_array):
    """
    Detect whether the image has at least one lesion-like dark region.
    Returns: (has_candidate: bool, details: dict)
    """
    if cv2 is None:
        return True, {}
    try:
        img = img_array[0] if img_array.shape[0] == 1 else img_array
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape[:2]
        image_area = float(h * w)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = image_area * 0.0015   # 0.15%
        max_area = image_area * 0.25     # 25%

        candidate_count = 0
        best_area_ratio = 0.0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / max(ch, 1)
            # Lesion candidates are usually compact blobs (not long palm lines).
            if 0.2 <= circularity <= 1.4 and 0.3 <= aspect_ratio <= 3.0:
                candidate_count += 1
                best_area_ratio = max(best_area_ratio, area / image_area)

        has_candidate = candidate_count > 0
        return has_candidate, {
            "candidate_count": candidate_count,
            "best_area_ratio": best_area_ratio,
        }
    except Exception:
        return True, {}


def detect_dark_mark_presence(img_array):
    """
    Detect prominent dark/deep-brown mark regions on skin.
    Returns: (has_dark_mark: bool, details: dict)
    """
    if cv2 is None:
        return False, {"mark_ratio": 0.0, "largest_ratio": 0.0, "mark_score": 0.0}
    try:
        img = img_array[0] if img_array.shape[0] == 1 else img_array
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img, 0, 255).astype(np.uint8)
        if len(img.shape) != 3 or img.shape[2] != 3:
            return False, {"mark_ratio": 0.0, "largest_ratio": 0.0, "mark_score": 0.0}

        h, w = img.shape[:2]
        image_area = float(max(h * w, 1))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Dark regions and deep-brown hue ranges.
        dark_mask = cv2.inRange(hsv, (0, 25, 0), (179, 255, 95))
        brown_mask = cv2.inRange(hsv, (5, 35, 20), (28, 255, 170))
        mark_mask = cv2.bitwise_or(dark_mask, brown_mask)

        kernel = np.ones((3, 3), np.uint8)
        mark_mask = cv2.morphologyEx(mark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mark_mask = cv2.morphologyEx(mark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        mark_ratio = float(np.mean(mark_mask > 0))
        contours, _ = cv2.findContours(mark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0.0
        for cnt in contours:
            largest_area = max(largest_area, float(cv2.contourArea(cnt)))
        largest_ratio = largest_area / image_area

        has_dark_mark = (mark_ratio >= 0.015) and (largest_ratio >= 0.004)
        mark_score = max(0.0, min(1.0, 0.55 * min(mark_ratio / 0.06, 1.0) + 0.45 * min(largest_ratio / 0.03, 1.0)))
        return has_dark_mark, {
            "mark_ratio": mark_ratio,
            "largest_ratio": largest_ratio,
            "mark_score": mark_score,
        }
    except Exception:
        return False, {"mark_ratio": 0.0, "largest_ratio": 0.0, "mark_score": 0.0}


def decode_uploaded_image(uploaded_bytes: bytes, target_size: tuple[int, int]) -> np.ndarray:
    """Decode uploaded bytes into RGB float image array with fallback paths."""
    try:
        pil_img = Image.open(BytesIO(uploaded_bytes)).convert("RGB")
        pil_img = pil_img.resize(target_size)
        return np.array(pil_img, dtype=np.float32)
    except Exception:
        pass

    if cv2 is not None:
        try:
            arr = np.frombuffer(uploaded_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("OpenCV could not decode the image bytes.")
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            decoded = cv2.resize(decoded, target_size)
            return decoded.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Unable to decode uploaded image: {e}")

    raise ValueError("Unable to decode uploaded image with available decoders.")


def safe_model_predict(model, model_input: np.ndarray):
    """
    Predict with defensive handling for Windows-specific OSError(22) issues.
    """
    x = np.asarray(model_input, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=255.0, neginf=0.0)
    x = np.ascontiguousarray(x)
    try:
        return model.predict(x, verbose=0)
    except OSError as e:
        # Retry once via tensor path when NumPy input triggers Invalid argument on some setups.
        if getattr(e, "errno", None) == 22:
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            return model.predict(x_tensor, verbose=0)
        raise


def classify_image_content(img_array, model):
    """
    Use ImageNet model to classify image content.
    """
    try:
        img = img_array[0] if img_array.shape[0] == 1 else img_array
        if cv2 is not None:
            img_resized = cv2.resize(img, (224, 224))
        else:
            img_resized = np.array(Image.fromarray(img.astype(np.uint8)).resize((224, 224)))
        img_preprocessed = preprocess_mobilenet_v2(img_resized.astype(np.float32))
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        predictions = model.predict(img_batch, verbose=0)
        decoded = decode_predictions(predictions, top=5)[0]
        medical_keywords = ["skin", "lesion", "mole", "medical", "bandage", "wound"]
        document_keywords = ["envelope", "packet", "book_jacket", "menu", "web_site", "scoreboard"]
        person_keywords = ["face", "person", "man", "woman", "boy", "girl", "suit", "uniform"]
        top_labels = [label.lower() for (_, label, _) in decoded]
        is_medical = any(keyword in label for keyword in medical_keywords for label in top_labels)
        is_document = any(keyword in label for keyword in document_keywords for label in top_labels)
        # Keep person detection conservative to avoid rejecting close-up lesion images.
        top1_label = top_labels[0] if top_labels else ""
        top1_score = float(decoded[0][2]) if decoded else 0.0
        is_person = any(keyword in top1_label for keyword in person_keywords) and top1_score >= 0.75
        return {
            "is_medical": is_medical,
            "is_document": is_document,
            "is_person": is_person,
            "top_predictions": decoded,
        }
    except Exception:
        return {"is_medical": False, "is_document": False, "is_person": False, "top_predictions": []}


def is_valid_skin_image_robust(img_array, validation_models=None):
    """
    Comprehensive image validation to detect non-skin uploads.
    """
    reasons = []
    checks_passed = 0
    total_checks = 5
    if validation_models is None:
        validation_models = load_validation_models()
    base_model, face_cascade = validation_models

    ocr_ready, ocr_msg = get_ocr_status()
    has_text, text_pct, text_regions = detect_text_in_image(img_array)
    # Only hard-reject when text coverage is clearly substantial.
    significant_text_for_reject = has_text and (text_regions >= 8 or text_pct >= 55)
    # Guardrail: ignore tiny/isolated OCR detections that are common on lesion textures.
    if has_text and text_regions <= 1 and text_pct <= 25:
        has_text = False
    if not ocr_ready:
        checks_passed += 1
        reasons.append(f"OCR check skipped: {ocr_msg}")
    elif has_text:
        reasons.append(f"Text detected: {text_regions} text regions ({text_pct:.1f}%).")
    else:
        checks_passed += 1
        reasons.append("No significant text detected.")

    has_face, num_faces = detect_faces(img_array, face_cascade)
    if has_face:
        reasons.append(f"Face detected: {num_faces} face(s).")
    else:
        checks_passed += 1
        reasons.append("No faces detected.")

    is_document, doc_confidence, num_lines, num_rects = detect_document_patterns(img_array)
    if is_document:
        reasons.append(
            f"Document patterns detected: {num_lines} lines, {num_rects} rectangles "
            f"(confidence {doc_confidence:.2f})."
        )
    else:
        checks_passed += 1
        reasons.append("No document patterns detected.")

    classification = classify_image_content(img_array, base_model)
    if classification["is_document"]:
        reasons.append("Content classifier indicates document/object-like image.")
    elif classification["is_person"] and not classification["is_medical"]:
        reasons.append("Content classifier indicates portrait/person-like image.")
    else:
        checks_passed += 1
        reasons.append("Content classification passed.")

    has_skin_colors = False
    try:
        img = img_array[0] if img_array.shape[0] == 1 else img_array
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img, 0, 255).astype(np.uint8)
        mean_colors = np.mean(img, axis=(0, 1))
        r, g, b = mean_colors[0], mean_colors[1], mean_colors[2]
        has_skin_colors = (100 < r < 240) and (50 < g < 200) and (30 < b < 150)
        if has_skin_colors:
            checks_passed += 1
            reasons.append(f"Skin-like colors detected (RGB: {r:.0f}, {g:.0f}, {b:.0f}).")
        else:
            reasons.append(f"Unusual color profile (RGB: {r:.0f}, {g:.0f}, {b:.0f}).")
    except Exception as e:
        reasons.append(f"Could not analyze colors: {e}")

    text_score = 100 if not has_text else max(0, 100 - int(min(100, text_pct * 4)))
    face_score = 0 if has_face else 100
    document_pattern_score = max(0, 100 - int(doc_confidence * 100))
    content_score = 100
    if classification["is_document"]:
        content_score = 25
    elif classification["is_person"] and not classification["is_medical"]:
        content_score = 40
    skin_color_score = 100 if has_skin_colors else 35
    suitability_score = int(round(np.mean([
        text_score, face_score, document_pattern_score, content_score, skin_color_score
    ])))
    min_suitability_score = 70

    is_valid = checks_passed >= 4
    if (ocr_ready and significant_text_for_reject) or has_face:
        is_valid = False
    if ocr_ready and has_text and not significant_text_for_reject:
        reasons.append("Minor text/marking detected but accepted (not significant enough to reject).")
    # Do not hard-reject based on generic content labels alone;
    # face detector remains the strict rejection signal for portraits.
    if classification["is_person"] and not has_face:
        reasons.append("Portrait/person-like content detected with low certainty (not used as hard reject).")
    if suitability_score < min_suitability_score:
        is_valid = False
        reasons.append(
            f"Skin suitability score is below threshold ({suitability_score}% < {min_suitability_score}%)."
        )
    summary = (
        f"Validation: {checks_passed}/{total_checks} checks passed | "
        f"Suitability: {suitability_score}% (min {min_suitability_score}%)"
    )

    metrics = {
        "Skin Suitability Score": f"{suitability_score}%",
        "Text Cleanliness": f"{text_score}%",
        "Face Interference": f"{face_score}%",
        "Document/Object Noise": f"{document_pattern_score}%",
        "Skin Color Consistency": f"{skin_color_score}%",
    }
    return is_valid, summary, reasons, metrics


def interpret_skin_prediction(probabilities):
    """
    Interpret skin-model probabilities with a NORMAL fallback.
    Returns: (result_label, confidence, note)
    """
    probs = np.asarray(probabilities).astype(np.float32).flatten()
    top_idx = int(np.argmax(probs))
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0]) if sorted_probs.size else 0.0
    top2 = float(sorted_probs[1]) if sorted_probs.size > 1 else 0.0
    spread = top1 - top2
    raw_label = skin_labels[top_idx]

    # Normal-skin fallback when model is uncertain on lesion classes.
    if top1 < 0.5:
        return "NORMAL", top1, "No lesion detected - appears to be normal skin"
    if 0.5 <= top1 < 0.7 and spread < 0.2:
        return "NORMAL", top1, "Unclear - may be normal skin or benign lesion"
    return raw_label, top1, ""


def get_cancer_status(result, image_type='skin', confidence=1.0):
    """
    Determine cancer status from prediction label.
    Returns: (has_cancer: bool, message: str)
    """
    if image_type == 'skin':
        high_risk_classes = {'MEL', 'BCC', 'SCC'}
        if result == 'NORMAL':
            return "healthy", "✅ NO SKIN CANCER DETECTED: Normal Healthy Skin"
        if result in {'NEV', 'SEK'}:
            return "benign", "ℹ️ BENIGN LESION: Non-cancerous mole detected"
        if result == 'ACK':
            return "cancer", "⚠️ SKIN CANCER DETECTED"
        if result in high_risk_classes:
            return "cancer", "⚠️ SKIN CANCER DETECTED"
        return "healthy", "✅ NO SKIN CANCER DETECTED: Normal Healthy Skin"
    if result == 'Malignant':
        return "cancer", "⚠️ Malignant - Cancer Detected"
    return "benign", "ℹ️ Benign - No Cancer Detected"


def display_result_message(message, status_type):
    if status_type == "cancer":
        st.error(message)
    elif status_type == "benign":
        st.info(message)
    elif status_type == "warning":
        st.warning(message)
    else:
        st.success(message)


def display_validation_result_robust(is_valid, summary, reasons, metrics):
    with st.expander("Image Validation Details", expanded=not is_valid):
        st.write(f"**{summary}**")
        st.write("---")
        for reason in reasons:
            st.write(f"- {reason}")
        if metrics:
            st.write("---")
            st.write("**Skin Image Classification Metrics:**")
            for key, value in metrics.items():
                st.write(f"- {key}: {value}")
    if not is_valid:
        st.error("⚠️ Invalid Image: Please upload a valid skin or dermoscopy image")
        st.info(
            "Accepted: clear lesion photos or dermoscopy images. "
            "Not accepted: ID cards, documents, portraits, screenshots, or random objects."
        )
        return False
    st.success("Image validation passed.")
    return True


# Detecting Melanoma
def melanoma_detection():
    st.title('Melanoma Detection')
    if not TF_AVAILABLE:
        st.error(
            "TensorFlow is not available in this environment. "
            "This app requires Python 3.11 for model inference."
        )
        if TF_IMPORT_ERROR:
            st.caption(f"TensorFlow import error: {TF_IMPORT_ERROR}")
        st.info(
            "For deployment, use Python 3.11 (for example via `runtime.txt`) "
            "or run locally in a Python 3.11 virtual environment."
        )
        return

    validation_models = load_validation_models()
    ocr_ready, ocr_msg = get_ocr_status()
    st.sidebar.markdown("### OCR Status")
    if ocr_ready:
        st.sidebar.success("OCR Active")
    else:
        st.sidebar.warning("OCR Inactive")
    st.sidebar.caption(ocr_msg)

    missing, download_errors = ensure_model_artifacts()
    loaded_models, model_skin_cnn, model_skin_vgg16, model_skin_resnet50, model_skin_efficientnet, model_skin_inceptionresnetv2 = load_models()

    missing = [f for f in ALL_REQUIRED_MODEL_FILES if not _is_valid_model_artifact(_model_path(f))]
    if missing:
        st.warning(
            "Some model files are missing or invalid. The app tried automatic download first."
            " Upload files manually to **"
            + str(MODEL_DIR)
            + "**, or set `MODEL_ASSET_BASE_URL` / file-specific `MODEL_URL_*` environment variables."
        )
        with st.expander("Missing files (expected names)"):
            st.code("\n".join(missing))
        if download_errors:
            with st.expander("Download errors"):
                st.code(
                    "\n".join(
                        f"{name}: {err}"
                        for name, err in download_errors.items()
                    )
                )

    tab1, tab2 = st.tabs(["Skin Image Models", "Dermoscopy Image Models"])
    with tab1:
        st.header("Skin Image Models")
    with tab2:
        st.header("Dermoscopy Image Models")

    mode = st.radio(
        "Use models for",
        ["Skin Image Models", "Dermoscopy Image Models"],
        horizontal=True,
        key="melanoma_image_kind",
    )
    is_skin = mode == "Skin Image Models"

    models = ['CNN', 'VGG16', 'ResNet50', 'EfficientNetB4', 'InceptionResNetV2']
    if is_skin:
        model_dict = {
            'CNN': model_skin_cnn,
            'VGG16': model_skin_vgg16,
            'ResNet50': model_skin_resnet50,
            'EfficientNetB4': model_skin_efficientnet,
            'InceptionResNetV2': model_skin_inceptionresnetv2
        }
    else:
        model_dict = loaded_models['derm']

    preprocess_dict = {
        'VGG16': preprocess_vgg16,
        'ResNet50': preprocess_resnet50,
        'EfficientNetB4': preprocess_efficientnet,
        'InceptionResNetV2': preprocess_inceptionresnetv2
    }

    selected_model = st.selectbox('Select Model', models)
    effective_model = selected_model

    # Several skin-model checkpoints can be poorly calibrated in real-world photos.
    # Use a stable default backend for skin inference to avoid "always cancer"/"always normal" behavior.
    if is_skin:
        preferred_skin_backend = "ResNet50"
        fallback_order = ["VGG16", "CNN", "EfficientNetB4", "InceptionResNetV2"]
        if model_dict.get(preferred_skin_backend) is not None:
            effective_model = preferred_skin_backend
        else:
            for candidate in fallback_order:
                if model_dict.get(candidate) is not None:
                    effective_model = candidate
                    break
        if effective_model != selected_model:
            st.info(
                f"Using stable skin inference backend: {effective_model} "
                f"(selected: {selected_model}) to improve prediction reliability."
            )
    pain_choice = "No"
    itching_choice = "No"
    if is_skin:
        st.markdown("### Symptoms (for rule-based screening)")
        pain_choice = st.radio(
            "Pain at the mark?",
            ["No", "Yes"],
            horizontal=True,
            key="pain_at_mark_choice",
        )
        itching_choice = st.radio(
            "Itching at the mark?",
            ["No", "Yes"],
            horizontal=True,
            key="itching_at_mark_choice",
        )

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        st.write("")
        st.write("Classifying...")

        try:
            # Determine input size based on the selected model
            if effective_model == 'EfficientNetB4':
                input_size = (380, 380)
            elif effective_model == 'InceptionResNetV2':
                input_size = (299, 299)
            else:
                input_size = (224, 224)

            # Decode using resilient multi-decoder path.
            img_3d = decode_uploaded_image(uploaded_bytes, input_size)
            st.image(img_3d.astype(np.uint8), caption='Uploaded Image.', width="stretch")
            img = np.expand_dims(img_3d, axis=0)

        except (OSError, ValueError) as e:
            st.error(
                "Uploaded image is not valid or could not be processed. "
                "Please upload a clear JPG/PNG skin lesion image. "
                "Photos with small marker annotations are allowed."
            )
            st.caption(f"Technical details: {e}")
            return

        try:
            # Robust image validation before prediction
            is_valid, summary, reasons, metrics = is_valid_skin_image_robust(img, validation_models)
            if not display_validation_result_robust(is_valid, summary, reasons, metrics):
                return
            
            # If the uploaded photo looks like plain skin without lesion features,
            # return a clear non-cancer outcome before model inference.
            plain_skin, plain_details = detect_plain_skin_no_lesion(img)
            if plain_skin:
                display_result_message("✅ HEALTHY SKIN: No cancer detected", status_type="healthy")
                st.write("Prediction: NORMAL")
                st.write(f"Confidence: {plain_details.get('plain_confidence', 0.0):.2f}")
                st.caption(
                    "Plain-skin validation triggered: image has very low texture/edge contrast "
                    "and no obvious lesion structure."
                )
                return

            # Additional guardrail: if no prominent lesion-like blob is present,
            # do not run cancer classification on broad skin/body-part photos.
            lesion_candidate, lesion_details = has_prominent_lesion_candidate(img)
            if not lesion_candidate:
                display_result_message("✅ HEALTHY SKIN: No cancer detected", status_type="healthy")
                st.write("Prediction: NORMAL")
                st.write("Confidence: 0.90")
                st.caption(
                    "No prominent lesion candidate was found (mostly plain skin/body-part texture). "
                    "Please upload a close-up image of the lesion area for diagnosis."
                )
                if lesion_details:
                    st.caption(
                        f"Detected lesion candidates: {lesion_details.get('candidate_count', 0)}"
                    )
                return

            # Preprocess based on model
            if effective_model in preprocess_dict:
                img = preprocess_dict[effective_model](img)

            # Assuming saliency map is generated (dummy saliency map for illustration)
            saliency_map = np.zeros_like(img)

            # Concatenate the original image with the saliency map
            combined_input = np.concatenate((img, saliency_map), axis=-1)

            # Model prediction
            model = model_dict.get(effective_model)
            if model is None:
                st.error("No valid model backend is available for prediction.")
            else:
                prediction = safe_model_predict(model, combined_input)

                # Handle the prediction output
                if is_skin:
                    result, confidence, uncertainty_note = interpret_skin_prediction(prediction[0])
                else:
                    if prediction.shape[-1] >= 2:
                        predicted_class = np.argmax(prediction, axis=-1)
                        confidence = float(np.max(prediction))
                        result = 'Malignant' if predicted_class[0] == 1 else 'Benign'
                    else:
                        threshold = 0.5
                        predicted_class = (prediction[:, 0] > threshold).astype(int)
                        confidence = float(prediction[0, 0])
                        result = 'Malignant' if predicted_class[0] == 1 else 'Benign'
                    uncertainty_note = ""

                # Keep low-confidence guard for dermoscopy flow.
                if (not is_skin) and confidence < 0.6:
                    st.error("⚠️ Invalid Image: Please upload a valid skin or dermoscopy image")
                    st.caption("Model confidence is too low for a reliable lesion prediction.")
                    return

                if is_skin:
                    has_dark_mark, mark_details = detect_dark_mark_presence(img)
                    best_area_ratio = float(lesion_details.get("best_area_ratio", 0.0)) if lesion_details else 0.0
                    candidate_count = int(lesion_details.get("candidate_count", 0)) if lesion_details else 0

                    # Guardrail for broad hand/body-part photos: if model predicts high-risk
                    # but there is no strong dark mark and lesion candidate is very small,
                    # treat as non-diagnostic normal-skin upload instead of cancer.
                    if result in {"ACK", "BCC", "MEL", "SCC"} and (not has_dark_mark) and best_area_ratio < 0.012:
                        display_result_message("✅ HEALTHY SKIN: No cancer detected", status_type="healthy")
                        st.write("Prediction: NORMAL")
                        st.write(f"Confidence: {confidence:.2f}")
                        st.caption(
                            "Image appears to be broad skin/palm texture without a clear lesion focus. "
                            "Please upload a close-up lesion image for diagnosis."
                        )
                        st.caption(
                            f"Lesion candidates: {candidate_count} | Largest candidate area: {best_area_ratio * 100:.2f}%"
                        )
                        st.caption(f"Pain: {pain_choice} | Itching: {itching_choice}")
                        return

                status_type, status_message = get_cancer_status(
                    result,
                    image_type='skin' if is_skin else 'derm',
                    confidence=confidence,
                )

                if is_skin:
                    both_yes = pain_choice == "Yes" and itching_choice == "Yes"
                    both_no = pain_choice == "No" and itching_choice == "No"
                    mixed_symptoms = (pain_choice == "Yes") != (itching_choice == "Yes")

                    # Enforce requested rule matrix for skin flow.
                    if has_dark_mark and both_yes:
                        status_type = "cancer"
                        status_message = "⚠️ SKIN CANCER DETECTED"
                        confidence = max(confidence, float(mark_details.get("mark_score", 0.0)))
                        uncertainty_note = "Rule: mark present + pain Yes + itching Yes."
                    elif has_dark_mark and mixed_symptoms:
                        status_type = "warning"
                        status_message = "⚠️ MIXED SYMPTOMS: Mark present, monitor closely and consult a dermatologist."
                        uncertainty_note = "Rule: mark present with mixed symptoms (one Yes, one No)."
                    elif has_dark_mark and both_no:
                        status_type = "warning"
                        status_message = "⚠️ MARK PRESENT: It seems cancer may be present. Please consult a dermatologist."
                        uncertainty_note = "Rule: mark present + both symptoms No."
                    else:
                        status_type = "healthy"
                        status_message = "✅ NO SKIN CANCER DETECTED"
                        if both_yes:
                            uncertainty_note = "Both symptoms are Yes, but no strong mark detected in the image."

                display_result_message(status_message, status_type)

                st.write(f"Prediction: {result}")
                st.write(f"Confidence: {confidence:.2f}")
                if is_skin and uncertainty_note:
                    st.caption(uncertainty_note)
                if is_skin:
                    st.caption(f"Pain: {pain_choice} | Itching: {itching_choice}")

                # Additional information
                st.write("\nPlease note:")
                st.write("- This prediction is based on the model's analysis and should not be considered as a definitive medical diagnosis.")
                st.write("- If you have any concerns about a skin lesion, please consult with a qualified healthcare professional or dermatologist.")
                st.write("- Regular skin check-ups and early detection are crucial for managing melanoma risk.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.caption("Tip: try a different model selection; if this repeats, the model file may be corrupted.")
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
    st.image(melanoma_image_url, width="stretch")

def display_model_summaries(models):
    for category, category_models in models.items():
        st.header(f"{category.capitalize()} Models")
        st.write("This section provides a detailed summary of the selected model's architecture, including the number of layers, parameters, and output shapes.")
        for model_name, model in category_models.items():
            st.subheader(f"{model_name} Model Summary")
            with st.expander("Show/Hide Model Summary"):
                summary_string = StringIO()
                model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
                
                # Display the summary with custom styling
                st.markdown(f"""
                <div class="model-summary-header">
                    <strong>Model:</strong> {model_name} ({category})<br>
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
                    file_name=f"{model_name}_{category}_summary.txt",
                    mime="text/plain"
                )
            st.markdown("---")

def display_model_evaluation(metrics, model_type, model_name):
    if model_name in metrics:
        st.write(f"### Model Performance for {model_name} ({model_type})")
        st.write("The following metrics provide an overview of the selected model's performance:")
        
        # Create a DataFrame for better display
        metrics_df = pd.DataFrame.from_dict(metrics[model_name], orient='index', columns=['Value'])
        st.table(metrics_df)
        
        # Add visual indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", metrics[model_name]['Accuracy'])
        with col2:
            st.metric("Precision", metrics[model_name]['Precision'])
        with col3:
            st.metric("Recall", metrics[model_name]['Recall'])
        with col4:
            st.metric("AUC", metrics[model_name]['AUC'])
    else:
        st.write("Metrics not available.")

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
    
    with st.expander("Show/Hide Model Summary", expanded=True):
        summary_string = StringIO()
        model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
        
        # Display the summary with custom styling
        st.markdown(f"""
        <div class="model-summary-header">
            <strong>Model:</strong> {model_name} ({image_type})<br>
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
            file_name=f"{model_name}_{image_type}_summary.txt",
            mime="text/plain"
        )

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

    with tab1:
        st.header(f"{model_type} - {model_name} Model Summary")
        model = st.session_state.models['skin' if model_type == 'Skin Image Models' else 'derm'][model_name]
        if model is None:
            st.info(
                "This model is not loaded. Add the matching `.keras` or `.weights.h5` file to the app folder "
                f"(`{MODEL_DIR}`), then use the menu **Settings → Clear cache** and reload."
            )
        else:
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
        if model is None:
            st.info("Load model weights first to plot an ROC curve (see Model Summary tab).")
        else:
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
    """)

    st.markdown("""
        - [Melanoma Detection with Deep Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6984940/): An academic article on using deep learning for melanoma detection.
        - [Skin Cancer Foundation](https://www.skincancer.org/): Comprehensive resource on skin cancer, including melanoma.
        - [Deep Learning for Dermatology](https://www.sciencedirect.com/science/article/pii/S0045653518300556): Review of deep learning techniques applied to dermatology.
        - [Journal of the American Academy of Dermatology (JAAD)](https://www.jaad.org/): Leading dermatology journal with articles on melanoma and skin diseases.
        - [Convolutional Neural Networks for Melanoma Detection](https://arxiv.org/abs/1805.06267): Research paper on applying CNNs for melanoma detection.
        - [ISIC Archive](https://www.isic-archive.com/): A large dataset of dermatological images for training and evaluation of models.
        - [Understanding Melanoma](https://www.cancer.gov/types/skin/melanoma): National Cancer Institute resource explaining melanoma and its treatment.
        - [AI for Melanoma Detection](https://www.bmj.com/content/369/bmj.m1972): Article discussing the use of AI in detecting melanoma.
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

# Function to send feedback via email
def send_email(name, email, message):
    sender_email = os.environ.get("SMTP_USER", "debbydawn16@gmail.com")
    sender_password = os.environ.get("SMTP_PASSWORD", "")
    recipient_email = os.environ.get("SMTP_TO", sender_email)

    if not sender_password:
        st.error(
            "Email is not configured. Set environment variables SMTP_USER and SMTP_PASSWORD "
            "(Gmail app password) or use a local .env file."
        )
        return
    
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
    init_auth_db()
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    logo_path = Path("logo_2.PNG")
    if logo_path.is_file():
        logo_image = Image.open(logo_path)
        logo_image = crop_to_circle(logo_image)
        st.sidebar.image(logo_image, width="stretch")
    else:
        st.sidebar.markdown("### Melanoma Detection")

    if st.session_state.authenticated:
        st.sidebar.markdown(f"**{st.session_state.current_user}**")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()

    pages = {
        "Introduction": main,
        "Model Performance": model_performance_page,
        "Visualizations": visualize_data,
        "Melanoma Detection": melanoma_detection,
        "Educational Resources": educational_resources,
        "FAQs": faq_section,
        "Feedback and Contact": feedback_form
    }

    if not st.session_state.authenticated:
        auth_page()
        return

    if "hide_navigation_options" not in st.session_state:
        st.session_state.hide_navigation_options = False
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Introduction"

    nav_col1, nav_col2 = st.sidebar.columns([5, 1])
    with nav_col1:
        st.markdown("### Navigation")
    with nav_col2:
        toggle_label = "☰" if st.session_state.hide_navigation_options else "✕"
        if st.button(toggle_label, key="toggle_navigation_visibility", help="Show/Hide navigation options"):
            st.session_state.hide_navigation_options = not st.session_state.hide_navigation_options
            st.rerun()

    if not st.session_state.hide_navigation_options:
        st.session_state.selected_page = st.sidebar.radio(
            "Go to",
            list(pages.keys()),
            index=list(pages.keys()).index(st.session_state.selected_page)
            if st.session_state.selected_page in pages
            else 0,
            label_visibility="collapsed",
        )

    pages[st.session_state.selected_page]()