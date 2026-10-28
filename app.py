from pathlib import Path
from datetime import datetime
import json
import csv
import hashlib
import io
import os
import shutil
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

try:
    import torch
    from torchvision import transforms
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
except ImportError as exc:
    st.error(f"Missing required package: {exc}. Install the requirements shown in the setup instructions.")
    st.stop()


# ============================================================
# ZERO LABEL VISION - EXPANDED SINGLE-FILE STREAMLIT APP
# Works with model.py, train.py, models/zerolabel_model.pth,
# data/gallery/<class_name>/ and data/test/<class_name>/
# ============================================================

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
GALLERY_DIR = DATA_DIR / "gallery"
TEST_DIR = DATA_DIR / "test"
UNLABELED_DIR = DATA_DIR / "unlabeled"
MODEL_PATH = APP_DIR / "models" / "zerolabel_model.pth"
RESULTS_DIR = APP_DIR / "results"
HISTORY_PATH = RESULTS_DIR / "search_history.csv"
REPORTS_DIR = RESULTS_DIR / "reports"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGE_SIZE = 224

for folder in (GALLERY_DIR, TEST_DIR, UNLABELED_DIR, RESULTS_DIR, REPORTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="ZeroLabel Vision",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


def apply_theme(theme_name):
    if theme_name == "Light":
        background = "#f5f7fb"
        surface = "#ffffff"
        text = "#172033"
        muted = "#596579"
        border = "#dce3ee"
        accent = "#2563eb"
        sidebar = "#eaf0f8"
    else:
        background = "#0b1120"
        surface = "#121c2e"
        text = "#edf3ff"
        muted = "#a6b5cc"
        border = "#263650"
        accent = "#60a5fa"
        sidebar = "#0f192b"

    st.markdown(
        f"""
        <style>
        :root {{
            --zv-bg: {background};
            --zv-surface: {surface};
            --zv-text: {text};
            --zv-muted: {muted};
            --zv-border: {border};
            --zv-accent: {accent};
        }}
        .stApp {{ background: var(--zv-bg); color: var(--zv-text); }}
        [data-testid="stSidebar"] {{ background: {sidebar}; border-right: 1px solid var(--zv-border); }}
        h1, h2, h3, h4, p, label, li, span {{ color: var(--zv-text); }}
        .zv-hero {{
            padding: 1.6rem 1.8rem; border: 1px solid var(--zv-border);
            border-radius: 22px; margin-bottom: 1.1rem;
            background: linear-gradient(135deg, {surface}, {background});
        }}
        .zv-hero h1 {{ font-size: 2.35rem; margin: 0; }}
        .zv-hero p {{ color: var(--zv-muted); font-size: 1.05rem; margin-bottom: 0; }}
        .zv-card {{
            background: var(--zv-surface); border: 1px solid var(--zv-border);
            padding: 1rem 1.15rem; border-radius: 16px; min-height: 118px;
        }}
        .zv-label {{ color: var(--zv-muted); font-size: .9rem; }}
        .zv-value {{ font-size: 1.8rem; font-weight: 750; margin-top: .2rem; }}
        .zv-note {{ color: var(--zv-muted); font-size: .85rem; }}
        div[data-testid="stMetric"] {{
            background: var(--zv-surface); border: 1px solid var(--zv-border);
            padding: 1rem; border-radius: 14px;
        }}
        div.stButton > button, div.stDownloadButton > button {{
            border-radius: 10px; font-weight: 650; min-height: 2.6rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: .5rem; }}
        .stTabs [data-baseweb="tab"] {{
            background: var(--zv-surface); border-radius: 8px 8px 0 0;
            padding: .6rem 1rem;
        }}
        hr {{ border-color: var(--zv-border); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme(st.session_state.theme)


def is_image_file(path):
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob("*") if is_image_file(p)])


def safe_open_image(path):
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def get_gallery_classes():
    if not GALLERY_DIR.exists():
        return []
    return sorted([p.name for p in GALLERY_DIR.iterdir() if p.is_dir()])


def get_dataset_summary(root):
    root = Path(root)
    classes = {}
    dimensions = []
    formats = {}
    total = 0
    invalid = 0
    if not root.exists():
        return {
            "total": 0, "classes": {}, "invalid": 0,
            "formats": {}, "dimensions": [], "class_count": 0
        }

    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        files = [p for p in class_dir.rglob("*") if is_image_file(p)]
        classes[class_dir.name] = len(files)
        for file_path in files:
            total += 1
            try:
                with Image.open(file_path) as image:
                    dimensions.append((image.width, image.height))
                    fmt = (image.format or file_path.suffix.lstrip(".")).upper()
                    formats[fmt] = formats.get(fmt, 0) + 1
                    image.verify()
            except Exception:
                invalid += 1

    # Include images placed directly in the root as an "unclassified" group.
    root_files = [p for p in root.iterdir() if is_image_file(p)]
    if root_files:
        classes["unclassified"] = len(root_files)
        total += len(root_files)
        for file_path in root_files:
            try:
                with Image.open(file_path) as image:
                    dimensions.append((image.width, image.height))
                    fmt = (image.format or file_path.suffix.lstrip(".")).upper()
                    formats[fmt] = formats.get(fmt, 0) + 1
            except Exception:
                invalid += 1

    return {
        "total": total,
        "classes": classes,
        "invalid": invalid,
        "formats": formats,
        "dimensions": dimensions,
        "class_count": len(classes),
    }


def load_history():
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=[
            "timestamp", "filename", "predicted_class", "similarity",
            "second_class", "second_similarity", "model_source"
        ])
    try:
        df = pd.read_csv(HISTORY_PATH)
        for col in ["timestamp", "filename", "predicted_class", "similarity",
                    "second_class", "second_similarity", "model_source"]:
            if col not in df.columns:
                df[col] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "timestamp", "filename", "predicted_class", "similarity",
            "second_class", "second_similarity", "model_source"
        ])


def append_history(record):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_history()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(HISTORY_PATH, index=False)
    st.session_state.search_history = df.to_dict(orient="records")


def make_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_path_text, modified_time):
    """Load the existing ZeroLabelModel and checkpoint when possible."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = make_transform()
    model = None
    status = "Feature fallback"
    error_text = ""

    try:
        from model import ZeroLabelModel
        model = ZeroLabelModel()
        if Path(model_path_text).exists():
            checkpoint = torch.load(model_path_text, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            else:
                state = checkpoint
            if isinstance(state, dict):
                cleaned = {}
                for key, value in state.items():
                    cleaned[key[7:] if key.startswith("module.") else key] = value
                missing, unexpected = model.load_state_dict(cleaned, strict=False)
                status = "Checkpoint loaded"
                if missing:
                    error_text = f"Some model parameters were not in the checkpoint ({len(missing)} missing)."
                if unexpected:
                    extra = f"{len(unexpected)} extra checkpoint parameters were ignored."
                    error_text = (error_text + " " + extra).strip()
            else:
                error_text = "Checkpoint format was not a state dictionary."
        else:
            error_text = "Model checkpoint not found. Using initialized model weights."
        model.to(device)
        model.eval()
    except Exception as exc:
        model = None
        error_text = f"Could not load model.py/checkpoint: {exc}"

    return model, transform, device, status, error_text


def image_to_tensor(image, transform):
    return transform(image.convert("RGB")).unsqueeze(0)


def feature_vector(image, model, transform, device):
    """Use the project's extract_features method when available."""
    tensor = image_to_tensor(image, transform).to(device)
    with torch.no_grad():
        if model is not None and hasattr(model, "extract_features"):
            features = model.extract_features(tensor)
        elif model is not None and hasattr(model, "backbone"):
            features = model.backbone(tensor)
        else:
            # Deterministic color/thumbnail feature fallback; not a trained semantic model.
            small = image.convert("RGB").resize((32, 32))
            arr = np.asarray(small, dtype=np.float32) / 255.0
            hist_features = []
            for channel in range(3):
                hist, _ = np.histogram(arr[:, :, channel], bins=32, range=(0, 1), density=True)
                hist_features.extend(hist.tolist())
            features = torch.tensor(hist_features, dtype=torch.float32).unsqueeze(0)
    if isinstance(features, (tuple, list)):
        features = features[0]
    if not torch.is_tensor(features):
        features = torch.as_tensor(features)
    features = features.detach().float().reshape(features.shape[0], -1)
    features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy()[0]


@st.cache_data(show_spinner=False)
def cached_gallery_features(paths_and_mtimes, model_signature):
    # This function is deliberately only a cache key holder. Actual vectors are
    # built by get_gallery_index, so changed image paths/mtime invalidate results.
    return [item[0] for item in paths_and_mtimes]


def get_gallery_index(model, transform, device):
    rows = []
    for class_name in get_gallery_classes():
        class_dir = GALLERY_DIR / class_name
        for path in list_images(class_dir):
            image = safe_open_image(path)
            if image is None:
                continue
            try:
                vector = feature_vector(image, model, transform, device)
                rows.append({
                    "class_name": class_name,
                    "path": str(path),
                    "filename": path.name,
                    "vector": vector,
                })
            except Exception:
                continue
    return rows


def classify_image(image, gallery_index):
    if not gallery_index:
        return []
    query = np.asarray(feature_vector(image, model, transform, device), dtype=np.float32).reshape(1, -1)
    scored = []
    for item in gallery_index:
        gallery_vec = np.asarray(item["vector"], dtype=np.float32).reshape(1, -1)
        if gallery_vec.shape[1] != query.shape[1]:
            continue
        score = float(cosine_similarity(query, gallery_vec)[0][0])
        scored.append({
            "class_name": item["class_name"],
            "filename": item["filename"],
            "path": item["path"],
            "similarity": score,
        })
    scored.sort(key=lambda row: row["similarity"], reverse=True)

    # Aggregate image-level matches by class, averaging each class's top few matches.
    class_scores = {}
    for row in scored:
        class_scores.setdefault(row["class_name"], []).append(row["similarity"])
    aggregated = []
    for class_name, values in class_scores.items():
        values.sort(reverse=True)
        top_values = values[: min(3, len(values))]
        aggregated.append({
            "class_name": class_name,
            "similarity": float(np.mean(top_values)),
            "matched_images": len(values),
        })
    aggregated.sort(key=lambda row: row["similarity"], reverse=True)
    # Attach a representative gallery filename to every class result. Older
    # UI sections expect ``filename``; the newer sections use
    # ``best_gallery_file``. Keep both keys for backward compatibility.
    for result in aggregated:
        representative = next(
            (row.get("filename", "") for row in scored
             if row.get("class_name") == result.get("class_name")),
            "",
        )
        result["filename"] = representative
        result["best_gallery_file"] = representative
    return aggregated


def image_hash(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()[:16]


def make_report_bytes(dataframe, file_type):
    if file_type == "JSON":
        return dataframe.to_json(orient="records", indent=2).encode("utf-8")
    return dataframe.to_csv(index=False).encode("utf-8")


def render_metric(label, value, note=""):
    st.markdown(
        f"""
        <div class="zv-card">
            <div class="zv-label">{label}</div>
            <div class="zv-value">{value}</div>
            <div class="zv-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_uploaded_image(uploaded_file, target_folder, class_name=None):
    safe_name = Path(uploaded_file.name).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported image format.")
    destination_dir = Path(target_folder)
    if class_name:
        destination_dir = destination_dir / class_name
    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(safe_name).stem
    destination = destination_dir / safe_name
    counter = 1
    while destination.exists():
        destination = destination_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    image = Image.open(uploaded_file).convert("RGB")
    image.save(destination)
    return destination


# Model load
model_mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0
model, transform, device, model_status, model_message = load_model_bundle(str(MODEL_PATH), model_mtime)

# Navigation and global controls
with st.sidebar:
    st.markdown("## 🔎 ZeroLabel Vision")
    st.caption("Self-supervised object recognition")
    st.selectbox("Interface theme", ["Dark", "Light"], key="theme")
    apply_theme(st.session_state.theme)
    st.divider()
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Image Recognition",
            "Gallery Management",
            "Image Comparison",
            "Viewing Angle Comparison",
            "Interactive Image Rotation",
            "Viewpoint Robustness Testing",
            "Top-K Predictions & Unknown Detection",
            "Prediction Score Explanation",
            "Similarity Heatmap",
            "Model Comparison",
            "Gallery Quality Checker",
            "Live Camera Recognition",
            "Dataset Analysis",
            "Evaluation Dashboard",
            "Search History",
            "Reports & Downloads",
            "Model Information",
        ],
    )
    st.divider()
    st.caption("Project workspace")
    st.code(str(APP_DIR), language="text")
    st.caption(f"Device: {device}")
    if MODEL_PATH.exists():
        st.success("Model checkpoint found")
    else:
        st.warning("Checkpoint not found")


gallery_index = get_gallery_index(model, transform, device)
gallery_classes = get_gallery_classes()
gallery_count = len(gallery_index)
gallery_summary = get_dataset_summary(GALLERY_DIR)
test_summary = get_dataset_summary(TEST_DIR)
unlabeled_summary = get_dataset_summary(UNLABELED_DIR)
history_df = load_history()


def page_header(title, subtitle):
    st.markdown(
        f'<div class="zv-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def make_class_chart(class_counts, title):
    if not class_counts:
        st.info("No class folders or images were found here yet.")
        return
    chart_df = pd.DataFrame({"Class": list(class_counts.keys()), "Images": list(class_counts.values())})
    chart_df = chart_df.sort_values("Images", ascending=False)
    fig, ax = plt.subplots(figsize=(9, max(3, len(chart_df) * 0.45)))
    ax.barh(chart_df["Class"], chart_df["Images"])
    ax.set_title(title)
    ax.set_xlabel("Number of images")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


if page == "Dashboard":
    page_header("ZeroLabel Vision", "An expanded workspace for gallery-based visual recognition and dataset evaluation.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("Gallery images", f"{gallery_summary['total']:,}", "Reference images")
    with c2:
        render_metric("Object classes", f"{len(gallery_classes):,}", "Gallery class folders")
    with c3:
        render_metric("Test images", f"{test_summary['total']:,}", "Labelled test dataset")
    with c4:
        render_metric("Searches recorded", f"{len(history_df):,}", "Persistent local history")
    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Gallery distribution")
        make_class_chart(gallery_summary["classes"], "Images in each gallery class")
    with right:
        st.subheader("System status")
        st.write(f"**Model:** {model_status}")
        st.write(f"**Compute device:** {device}")
        st.write(f"**Gallery vectors:** {gallery_count}")
        st.write(f"**Gallery location:** `{GALLERY_DIR}`")
        st.write(f"**Test location:** `{TEST_DIR}`")
        if model_message:
            st.warning(model_message)
        if gallery_count == 0:
            st.info("Add reference images in data/gallery/<class_name>/ or use Gallery Management.")
    st.subheader("Recent recognition searches")
    if history_df.empty:
        st.info("No searches recorded yet. Open Image Recognition to run your first search.")
    else:
        st.dataframe(history_df.tail(8).iloc[::-1], use_container_width=True, hide_index=True)
    st.caption("Similarity is a feature-space match score, not a probability of correctness.")


elif page == "Image Recognition":
    page_header("Image Recognition", "Upload one image and compare its visual features with your reference gallery.")
    if not gallery_index:
        st.warning("Your gallery is empty or its images could not be read.")
        st.code("data/gallery/bottle/example.jpg\ndata/gallery/cup/example.jpg\ndata/gallery/phone/example.jpg")
    uploaded = st.file_uploader("Upload an image to recognize", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"], key="recognition_upload")
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        col_img, col_result = st.columns([1, 1.15])
        with col_img:
            st.image(image, caption=uploaded.name, use_container_width=True)
            st.caption(f"Image size: {image.width} × {image.height}")
            st.caption(f"Image fingerprint: {image_hash(image)}")
        with col_result:
            if st.button("Recognize image", type="primary", use_container_width=True, disabled=not gallery_index):
                with st.spinner("Extracting features and searching the gallery..."):
                    predictions = classify_image(image, gallery_index)
                if not predictions:
                    st.error("Could not compare this image with the current gallery. Check model feature dimensions.")
                else:
                    best = predictions[0]
                    second = predictions[1] if len(predictions) > 1 else None
                    st.session_state.last_prediction = {
                        "filename": uploaded.name,
                        "image_hash": image_hash(image),
                        "predictions": predictions,
                    }
                    record = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "filename": uploaded.name,
                        "predicted_class": best["class_name"],
                        "similarity": round(best["similarity"], 6),
                        "second_class": second["class_name"] if second else "",
                        "second_similarity": round(second["similarity"], 6) if second else "",
                        "model_source": model_status,
                    }
                    append_history(record)
                    st.success(f"Closest gallery class: {best['class_name']}")
                    st.metric("Top similarity", f"{best['similarity']:.4f}")
                    if second:
                        st.write(f"Second closest: **{second['class_name']}** ({second['similarity']:.4f})")
                    st.caption("The score ranks visual feature similarity. It is not a calibrated confidence probability.")
            if st.session_state.last_prediction:
                saved = st.session_state.last_prediction
                st.subheader("Latest result")
                if saved.get("filename") == uploaded.name:
                    for i, prediction in enumerate(saved["predictions"][:5], start=1):
                        st.write(f"{i}. **{prediction['class_name']}** — {prediction['similarity']:.4f}")
                        st.progress(float(np.clip((prediction["similarity"] + 1) / 2, 0, 1)))
    st.subheader("Reference gallery preview")
    if gallery_index:
        preview_cols = st.columns(min(4, len(gallery_index)))
        for i, item in enumerate(gallery_index[:8]):
            with preview_cols[i % len(preview_cols)]:
                preview = safe_open_image(item["path"])
                if preview is not None:
                    st.image(preview, use_container_width=True)
                    st.caption(f"{item['class_name']} · {item['filename']}")


elif page == "Gallery Management":
    page_header("Gallery Management", "Organize reference images used for nearest-neighbour recognition.")
    classes = get_gallery_classes()
    tab_add, tab_browse, tab_remove = st.tabs(["Add images", "Browse gallery", "Remove image"])
    with tab_add:
        st.write("Choose an existing class or create a new class folder.")
        existing_options = ["Create new class"] + classes
        chosen_class = st.selectbox("Gallery class", existing_options)
        if chosen_class == "Create new class":
            class_name = st.text_input("New class name", placeholder="e.g. bottle")
        else:
            class_name = chosen_class
        files = st.file_uploader(
            "Choose one or more reference images",
            type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
            accept_multiple_files=True,
            key="gallery_add_upload",
        )
        if st.button("Add selected images", type="primary", disabled=not files or not class_name.strip()):
            clean_class = "".join(ch for ch in class_name.strip() if ch.isalnum() or ch in ("-", "_", " "))
            clean_class = clean_class.strip().replace(" ", "_")
            if not clean_class:
                st.error("Enter a valid class name.")
            else:
                saved_paths = []
                for file in files:
                    try:
                        saved_paths.append(save_uploaded_image(file, GALLERY_DIR, clean_class))
                    except Exception as exc:
                        st.warning(f"Could not save {file.name}: {exc}")
                if saved_paths:
                    st.success(f"Added {len(saved_paths)} image(s) to class '{clean_class}'.")
                    st.cache_resource.clear()
                    st.rerun()
    with tab_browse:
        classes = get_gallery_classes()
        if not classes:
            st.info("No class folders exist yet. Add images above.")
        else:
            selected = st.selectbox("Select class to browse", classes)
            paths = list_images(GALLERY_DIR / selected)
            st.write(f"**{selected}** — {len(paths)} image(s)")
            if paths:
                cols = st.columns(4)
                for i, path in enumerate(paths):
                    with cols[i % 4]:
                        image = safe_open_image(path)
                        if image is not None:
                            st.image(image, use_container_width=True)
                            st.caption(path.name)
    with tab_remove:
        all_paths = list_images(GALLERY_DIR)
        if not all_paths:
            st.info("There are no gallery images to remove.")
        else:
            path_options = {f"{p.parent.name}/{p.name}": p for p in all_paths}
            selected_path_label = st.selectbox("Choose image to remove", list(path_options.keys()))
            st.warning("This permanently deletes the selected file from your local gallery.")
            confirm_delete = st.checkbox("I confirm that I want to delete this image.")
            if st.button("Delete selected image", type="primary", disabled=not confirm_delete):
                target = path_options[selected_path_label]
                try:
                    target.unlink()
                    st.success(f"Deleted {selected_path_label}")
                    st.cache_resource.clear()
                    st.rerun()
                except OSError as exc:
                    st.error(f"Could not delete image: {exc}")


elif page == "Image Comparison":
    page_header("Image Comparison", "Compare two images directly using the same feature extractor.")
    col_a, col_b = st.columns(2)
    with col_a:
        file_a = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="compare_a")
    with col_b:
        file_b = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="compare_b")
    if file_a and file_b:
        image_a = Image.open(file_a).convert("RGB")
        image_b = Image.open(file_b).convert("RGB")
        show_a, show_b = st.columns(2)
        with show_a:
            st.image(image_a, caption=f"Image A: {file_a.name}", use_container_width=True)
        with show_b:
            st.image(image_b, caption=f"Image B: {file_b.name}", use_container_width=True)
        if st.button("Compare images", type="primary", use_container_width=True):
            try:
                vector_a = feature_vector(image_a, model, transform, device).reshape(1, -1)
                vector_b = feature_vector(image_b, model, transform, device).reshape(1, -1)
                if vector_a.shape[1] != vector_b.shape[1]:
                    st.error("The two feature vectors have different dimensions.")
                else:
                    similarity = float(cosine_similarity(vector_a, vector_b)[0][0])
                    st.metric("Cosine similarity", f"{similarity:.4f}")
                    st.progress(float(np.clip((similarity + 1) / 2, 0, 1)))
                    st.caption("Higher values indicate more similar feature vectors. This score is not a probability.")
                    comparison_df = pd.DataFrame([{
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "image_a": file_a.name,
                        "image_b": file_b.name,
                        "cosine_similarity": similarity,
                        "image_a_hash": image_hash(image_a),
                        "image_b_hash": image_hash(image_b),
                    }])
                    st.download_button(
                        "Download comparison CSV",
                        comparison_df.to_csv(index=False).encode("utf-8"),
                        file_name="image_comparison.csv",
                        mime="text/csv",
                    )
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")


elif page == "Viewing Angle Comparison":
    page_header(
        "Viewing Angle Comparison (0°, 45°, 90°)",
        "Compare recognition results after rotating one uploaded image by 0°, 45°, and 90° in the image plane.",
    )
    st.info(
        "Important: these are 2D image rotations, not real photographs of a 3D object from different camera viewpoints. "
        "This experiment measures sensitivity to in-plane rotation only. True viewpoint testing requires images of the same object captured from different camera positions."
    )
    if not gallery_index:
        st.warning("The reference gallery is empty or its images could not be read.")
        st.code("data/gallery/bottle/example.jpg\ndata/gallery/cup/example.jpg\ndata/gallery/phone/example.jpg")

    angle_upload = st.file_uploader(
        "Upload an object image",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
        key="view_angle_upload",
    )
    if angle_upload is not None:
        try:
            base_image = Image.open(angle_upload).convert("RGB")
            st.image(base_image, caption=f"Original image: {angle_upload.name}", width=320)
            st.caption("The same uploaded image will be rotated and evaluated against the current reference gallery.")

            if st.button("Run 0° / 45° / 90° comparison", type="primary", use_container_width=True, disabled=not gallery_index):
                angle_results = []
                rotated_images = {}
                progress = st.progress(0, text="Preparing angle comparison...")
                for index, angle in enumerate((0, 45, 90), start=1):
                    rotated = base_image.copy() if angle == 0 else base_image.rotate(
                        angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0)
                    )
                    rotated_images[angle] = rotated
                    try:
                        predictions = classify_image(rotated, gallery_index)
                    except Exception as exc:
                        st.error(f"Could not evaluate the {angle}° image: {exc}")
                        predictions = []

                    if predictions:
                        best = predictions[0]
                        second = predictions[1] if len(predictions) > 1 else None
                        angle_results.append({
                            "angle_degrees": angle,
                            "predicted_class": best["class_name"],
                            "top_similarity": round(float(best["similarity"]), 6),
                            "second_class": second["class_name"] if second else "",
                            "second_similarity": round(float(second["similarity"]), 6) if second else np.nan,
                            "model_source": model_status,
                        })
                        for item in predictions:
                            angle_results[-1][f"class_score::{item['class_name']}"] = round(float(item["similarity"]), 6)
                    progress.progress(index / 3, text=f"Evaluated rotation {angle}°")

                progress.empty()
                if not angle_results:
                    st.error("No results were produced. Check the model and gallery feature dimensions.")
                else:
                    result_df = pd.DataFrame(angle_results)
                    st.session_state["view_angle_results"] = result_df
                    st.session_state["view_angle_images"] = rotated_images
                    st.session_state["view_angle_filename"] = angle_upload.name

            if "view_angle_results" in st.session_state and st.session_state.get("view_angle_filename") == angle_upload.name:
                result_df = st.session_state["view_angle_results"]
                rotated_images = st.session_state.get("view_angle_images", {})
                st.subheader("Rotated image views")
                image_cols = st.columns(3)
                for col, angle in zip(image_cols, (0, 45, 90)):
                    with col:
                        if angle in rotated_images:
                            st.image(rotated_images[angle], caption=f"{angle}° rotation", use_container_width=True)

                st.subheader("Recognition result by angle")
                display_cols = ["angle_degrees", "predicted_class", "top_similarity", "second_class", "second_similarity"]
                st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)

                chart_df = result_df[["angle_degrees", "top_similarity"]].set_index("angle_degrees")
                st.line_chart(chart_df, y="top_similarity", x_label="Image-plane rotation (degrees)", y_label="Top class similarity")

                classes = [c for c in result_df.columns if c.startswith("class_score::")]
                if classes:
                    class_chart = result_df.set_index("angle_degrees")[classes].rename(
                        columns=lambda name: name.replace("class_score::", "")
                    )
                    st.subheader("Class similarity across rotations")
                    st.line_chart(class_chart, x_label="Image-plane rotation (degrees)", y_label="Class similarity")

                st.download_button(
                    "Download angle comparison CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="zerolabel_viewing_angle_comparison.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                if result_df["predicted_class"].nunique() == 1:
                    st.success("The top-ranked class stayed the same for all tested image-plane rotations.")
                else:
                    st.warning("The top-ranked class changed across rotations. Review the gallery and test with more images before drawing conclusions.")
                st.caption(
                    "Similarity scores are feature-space ranking scores, not probabilities. This single-image test is exploratory; "
                    "report accuracy across a labeled test set for a reliable performance claim."
                )
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"Could not open the uploaded image: {exc}")


elif page == "Dataset Analysis":
    page_header("Dataset Analysis", "Inspect gallery, labelled test images, and the unlabeled training set.")
    ds1, ds2, ds3 = st.tabs(["Gallery dataset", "Test dataset", "Unlabeled dataset"])
    summaries = [
        ("Gallery", gallery_summary, GALLERY_DIR),
        ("Test", test_summary, TEST_DIR),
        ("Unlabeled", unlabeled_summary, UNLABELED_DIR),
    ]
    for tab, (name, summary, folder) in zip([ds1, ds2, ds3], summaries):
        with tab:
            a, b, c = st.columns(3)
            a.metric("Images", summary["total"])
            b.metric("Class folders", summary["class_count"])
            c.metric("Unreadable images", summary["invalid"])
            st.caption(f"Folder: {folder}")
            make_class_chart(summary["classes"], f"{name} images per class")
            format_df = pd.DataFrame([
                {"Format": key, "Count": value}
                for key, value in summary["formats"].items()
            ])
            if not format_df.empty:
                st.subheader("Image formats")
                st.dataframe(format_df, use_container_width=True, hide_index=True)
            if summary["dimensions"]:
                dims = pd.DataFrame(summary["dimensions"], columns=["Width", "Height"])
                st.subheader("Image dimensions")
                d1, d2, d3 = st.columns(3)
                d1.metric("Average width", f"{dims['Width'].mean():.1f}px")
                d2.metric("Average height", f"{dims['Height'].mean():.1f}px")
                d3.metric("Unique sizes", dims.drop_duplicates().shape[0])
                st.dataframe(dims.describe().round(2), use_container_width=True)
            report_df = pd.DataFrame([
                {"dataset": name, "class_name": class_name, "image_count": count}
                for class_name, count in summary["classes"].items()
            ])
            st.download_button(
                f"Download {name.lower()} summary CSV",
                report_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{name.lower()}_dataset_summary.csv",
                mime="text/csv",
                key=f"download_{name}",
            )


elif page == "Evaluation Dashboard":
    page_header("Evaluation Dashboard", "Measure gallery-based classification against labelled images in data/test/<class_name>/.")
    st.info("Evaluation assumes each test image's parent folder is its true label. Use independent test images for meaningful results.")
    test_files = []
    for class_dir in sorted([p for p in TEST_DIR.iterdir() if p.is_dir()]):
        for path in list_images(class_dir):
            test_files.append((path, class_dir.name))
    if not gallery_index:
        st.warning("Add gallery images first.")
    elif not test_files:
        st.warning("No labelled test images found. Create folders such as data/test/bottle/ and place test images inside.")
    else:
        max_images = st.slider("Maximum test images to evaluate", min_value=1, max_value=max(1, min(len(test_files), 500)), value=min(len(test_files), 100))
        if st.button("Run evaluation", type="primary"):
            true_labels = []
            predicted_labels = []
            eval_rows = []
            progress = st.progress(0)
            status = st.empty()
            selected_test = test_files[:max_images]
            for index, (path, true_label) in enumerate(selected_test):
                image = safe_open_image(path)
                if image is None:
                    continue
                try:
                    predictions = classify_image(image, gallery_index)
                    if not predictions:
                        continue
                    predicted = predictions[0]["class_name"]
                    score = predictions[0]["similarity"]
                    true_labels.append(true_label)
                    predicted_labels.append(predicted)
                    eval_rows.append({
                        "filename": path.name,
                        "true_class": true_label,
                        "predicted_class": predicted,
                        "top_similarity": score,
                        "correct": true_label == predicted,
                        "path": str(path.relative_to(APP_DIR)),
                    })
                except Exception as exc:
                    eval_rows.append({
                        "filename": path.name, "true_class": true_label,
                        "predicted_class": "ERROR", "top_similarity": np.nan,
                        "correct": False, "path": str(path.relative_to(APP_DIR)),
                        "error": str(exc),
                    })
                progress.progress((index + 1) / len(selected_test))
                status.write(f"Evaluated {index + 1} of {len(selected_test)} images")
            progress.empty()
            status.empty()
            eval_df = pd.DataFrame(eval_rows)
            st.session_state["evaluation_results"] = eval_df
            if true_labels:
                accuracy = accuracy_score(true_labels, predicted_labels)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, predicted_labels, average="weighted", zero_division=0
                )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{accuracy:.2%}")
                m2.metric("Weighted precision", f"{precision:.2%}")
                m3.metric("Weighted recall", f"{recall:.2%}")
                m4.metric("Weighted F1", f"{f1:.2%}")
                labels = sorted(set(true_labels) | set(predicted_labels))
                cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
                fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.0), max(4, len(labels) * 0.8)))
                image_plot = ax.imshow(cm, interpolation="nearest")
                ax.set_title("Confusion matrix")
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_yticklabels(labels)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
                fig.colorbar(image_plot, ax=ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.error("No test images could be evaluated.")
            st.subheader("Per-image evaluation results")
            st.dataframe(eval_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download evaluation CSV",
                eval_df.to_csv(index=False).encode("utf-8"),
                file_name="zerolabel_evaluation.csv",
                mime="text/csv",
            )
    if "evaluation_results" in st.session_state:
        prior_eval = st.session_state["evaluation_results"]
        if not prior_eval.empty:
            st.subheader("Latest saved evaluation")
            st.dataframe(prior_eval, use_container_width=True, hide_index=True)
            st.download_button(
                "Download latest evaluation JSON",
                prior_eval.to_json(orient="records", indent=2).encode("utf-8"),
                file_name="zerolabel_evaluation.json",
                mime="application/json",
                key="eval_json_download",
            )


elif page == "Search History":
    page_header("Search History", "Review and export recognition searches stored on this computer.")
    history_df = load_history()
    top_a, top_b = st.columns([2, 1])
    with top_a:
        search_text = st.text_input("Filter by filename or predicted class", placeholder="Type bottle, cup, or a filename")
    with top_b:
        limit = st.selectbox("Rows to show", [25, 50, 100, "All"], index=1)
    filtered = history_df.copy()
    if search_text.strip():
        mask = (
            filtered["filename"].astype(str).str.contains(search_text, case=False, na=False)
            | filtered["predicted_class"].astype(str).str.contains(search_text, case=False, na=False)
        )
        filtered = filtered[mask]
    if limit != "All":
        filtered = filtered.tail(int(limit))
    st.metric("Matching records", len(filtered))
    if filtered.empty:
        st.info("No matching history records.")
    else:
        st.dataframe(filtered.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered history CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="search_history.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download filtered history JSON",
            filtered.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="search_history.json",
            mime="application/json",
        )
    if not history_df.empty:
        st.divider()
        if st.button("Clear all search history", type="secondary"):
            st.session_state["confirm_clear_history"] = True
        if st.session_state.get("confirm_clear_history", False):
            st.warning("This removes the local search_history.csv file.")
            confirm = st.checkbox("Confirm permanent history deletion")
            if st.button("Confirm clear history", disabled=not confirm):
                try:
                    HISTORY_PATH.unlink(missing_ok=True)
                    st.session_state["confirm_clear_history"] = False
                    st.success("Search history cleared.")
                    st.rerun()
                except OSError as exc:
                    st.error(f"Could not clear history: {exc}")


elif page == "Reports & Downloads":
    page_header("Reports & Downloads", "Export dataset summaries, search history, and the latest evaluation.")
    history_df = load_history()
    report_tabs = st.tabs(["Search history", "Dataset summary", "Evaluation"])
    with report_tabs[0]:
        st.write("Export all recorded recognition searches.")
        st.download_button(
            "Download search history CSV",
            history_df.to_csv(index=False).encode("utf-8"),
            file_name="zerolabel_search_history.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download search history JSON",
            history_df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="zerolabel_search_history.json",
            mime="application/json",
        )
    with report_tabs[1]:
        dataset_rows = []
        for dataset_name, summary in [
            ("gallery", gallery_summary),
            ("test", test_summary),
            ("unlabeled", unlabeled_summary),
        ]:
            for class_name, count in summary["classes"].items():
                dataset_rows.append({
                    "dataset": dataset_name,
                    "class_name": class_name,
                    "image_count": count,
                    "total_dataset_images": summary["total"],
                    "class_count": summary["class_count"],
                    "unreadable_images": summary["invalid"],
                })
        dataset_df = pd.DataFrame(dataset_rows)
        st.dataframe(dataset_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download combined dataset CSV",
            dataset_df.to_csv(index=False).encode("utf-8"),
            file_name="zerolabel_dataset_report.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download combined dataset JSON",
            dataset_df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="zerolabel_dataset_report.json",
            mime="application/json",
        )
    with report_tabs[2]:
        if "evaluation_results" in st.session_state and not st.session_state["evaluation_results"].empty:
            evaluation_df = st.session_state["evaluation_results"]
            st.dataframe(evaluation_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download evaluation report CSV",
                evaluation_df.to_csv(index=False).encode("utf-8"),
                file_name="zerolabel_evaluation_report.csv",
                mime="text/csv",
            )
        else:
            st.info("Run the Evaluation Dashboard first to generate an evaluation report.")
    st.caption(f"Report output folder: {REPORTS_DIR}")



elif page == "Interactive Image Rotation":
    page_header("Interactive Image Rotation", "Rotate a query image and see its nearest gallery matches update.")
    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"], key="interactive_rotate_upload")
    if upload:
        original = Image.open(upload).convert("RGB")
        angle = st.slider("Rotation angle", -180, 180, 0, 5)
        rotated = original.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0))
        c1, c2 = st.columns(2)
        c1.image(original, caption="Original", use_container_width=True)
        c2.image(rotated, caption=f"Rotated {angle}°", use_container_width=True)
        if gallery_index:
            preds = classify_image(rotated, gallery_index)
            if preds:
                st.subheader("Updated matches")
                for rank, pred in enumerate(preds[:5], 1):
                    st.write(f"{rank}. **{pred['class_name']}** — similarity {pred['similarity']:.4f}")
                st.caption("Similarity is a feature-space ranking score, not a probability.")
        else:
            st.warning("Add reference images to data/gallery first.")

elif page == "Viewpoint Robustness Testing":
    page_header("Viewpoint Robustness Testing", "Measure prediction stability across image-plane rotations.")
    st.warning("Rotating a 2D image is not the same as capturing a 3D object from a new camera viewpoint. Use separate real viewpoint photos for true viewpoint evaluation.")
    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="robustness_upload")
    angles = st.multiselect("Angles to test", [-90, -45, 0, 45, 90, 135, 180], default=[0, 45, 90])
    if upload and gallery_index and angles and st.button("Run robustness test", type="primary"):
        base = Image.open(upload).convert("RGB")
        rows = []
        for a in angles:
            img = base.rotate(a, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0,0,0))
            pred = classify_image(img, gallery_index)
            if pred:
                rows.append({"angle_degrees": a, "predicted_class": pred[0]["class_name"], "top_similarity": pred[0]["similarity"], "second_class": pred[1]["class_name"] if len(pred)>1 else ""})
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.metric("Prediction stability", f"{df['predicted_class'].value_counts().max()/len(df):.1%}", help="Share of tested angles assigned to the most frequent predicted class; not ground-truth accuracy.")
            st.line_chart(df.set_index("angle_degrees")["top_similarity"])
            st.download_button("Download robustness CSV", df.to_csv(index=False).encode(), "viewpoint_robustness.csv", "text/csv")
    elif upload and not gallery_index:
        st.warning("Add gallery images first.")

elif page == "Top-K Predictions & Unknown Detection":
    page_header("Top-K Predictions and Unknown Object Detection", "Inspect ranked matches and flag weak gallery matches.")
    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="topk_upload")
    k = st.selectbox("Number of results", [3, 5, 10], index=1)
    threshold = st.slider("Unknown-object similarity threshold", -1.0, 1.0, 0.35, 0.01, help="Set this threshold using validation data; it is not a probability cutoff.")
    if upload and gallery_index:
        img = Image.open(upload).convert("RGB")
        st.image(img, width=280)
        preds = classify_image(img, gallery_index)
        if preds:
            best = preds[0]
            if best["similarity"] < threshold:
                st.warning(f"Possibly unknown object: best similarity {best['similarity']:.4f} is below {threshold:.2f}.")
            else:
                st.success(f"Gallery match: {best['class_name']} (similarity {best['similarity']:.4f})")
            for i, row in enumerate(preds[:k], 1):
                st.write(f"{i}. **{row['class_name']}** — {row['similarity']:.4f} · {row['filename']}")
            st.caption("Similarity scores rank visual features and are not calibrated probabilities. Choose the unknown threshold using held-out known and unknown examples.")
            st.download_button("Export top-K predictions", pd.DataFrame(preds[:k]).to_csv(index=False).encode(), "top_k_predictions.csv", "text/csv")
    elif upload:
        st.warning("Add gallery images first.")

elif page == "Prediction Score Explanation":
    page_header("Prediction Score Explanation", "Understand what the nearest-gallery similarity score tells you and what it does not.")
    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="score_explain_upload")
    if upload and gallery_index:
        query_image = Image.open(upload).convert("RGB")
        preds = classify_image(query_image, gallery_index)
        if preds:
            best = preds[0]
            second = preds[1] if len(preds) > 1 else None
            a, b, c = st.columns(3)
            a.metric("Best-matching class", best["class_name"])
            b.metric("Top similarity", f"{best['similarity']:.4f}")
            c.metric("Gap to second result", f"{best['similarity']-second['similarity']:.4f}" if second else "N/A")
            st.write("**How to interpret this score**")
            st.markdown("""- The app extracts a feature vector from the query and compares it with gallery feature vectors using cosine similarity.
- Higher scores indicate a closer match in the current feature space; they do not guarantee the predicted class is correct.
- The score is **not a probability**, is not calibrated confidence, and should not be read as a percentage chance of correctness.
- Choose any unknown-object threshold by testing labelled known objects and genuinely unknown objects on held-out data.""")
            st.subheader("Top matches")
            st.dataframe(pd.DataFrame(preds[:10]), use_container_width=True, hide_index=True)
            st.bar_chart(pd.DataFrame(preds[:10]).set_index("class_name")["similarity"])
    elif upload:
        st.warning("Add gallery images first.")

elif page == "Similarity Heatmap":
    page_header("Similarity Heatmap", "Compare one query image against gallery images and aggregate class scores.")
    upload = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="heatmap_upload")
    if upload and gallery_index:
        query = Image.open(upload).convert("RGB")
        qv = np.asarray(feature_vector(query, model, transform, device), dtype=np.float32)
        rows = []
        for item in gallery_index:
            vec = np.asarray(item["vector"], dtype=np.float32)
            if vec.size == qv.size:
                rows.append({"class_name": item["class_name"], "filename": item["filename"], "similarity": float(np.dot(qv, vec)/(np.linalg.norm(qv)*np.linalg.norm(vec)+1e-12))})
        if rows:
            df = pd.DataFrame(rows)
            heat = df.pivot_table(index="class_name", columns="filename", values="similarity", aggfunc="max")
            fig, ax = plt.subplots(figsize=(max(7, heat.shape[1]*0.6), max(3, heat.shape[0]*0.7)))
            im = ax.imshow(heat.fillna(0).values, aspect="auto", cmap="viridis", vmin=-1, vmax=1)
            ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns, rotation=70, ha="right", fontsize=8)
            ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index)
            ax.set_title("Query-to-gallery cosine similarity")
            fig.colorbar(im, ax=ax, label="Cosine similarity"); fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.dataframe(df.sort_values("similarity", ascending=False), use_container_width=True, hide_index=True)
            st.download_button("Download similarity scores", df.to_csv(index=False).encode(), "similarity_heatmap_scores.csv", "text/csv")
    elif upload:
        st.warning("Add gallery images first.")

elif page == "Model Comparison":
    page_header("Model Comparison", "Compare the current feature extractor with a simple color-histogram baseline.")
    st.info("The baseline is a handcrafted color descriptor, not a second trained neural network. This comparison is meaningful only as a feature-method comparison.")
    upload = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="model_compare_upload")
    if upload and gallery_index:
        query = Image.open(upload).convert("RGB")
        def color_descriptor(im):
            arr = np.asarray(im.convert("RGB").resize((32,32)), dtype=np.float32)/255.0
            vals=[]
            for ch in range(3):
                h,_=np.histogram(arr[:,:,ch], bins=32, range=(0,1), density=True); vals.extend(h.tolist())
            v=np.asarray(vals,dtype=np.float32); return v/(np.linalg.norm(v)+1e-12)
        query_current = np.asarray(feature_vector(query, model, transform, device), dtype=np.float32)
        query_color = color_descriptor(query)
        result_rows=[]
        for method, qv in [("Current feature extractor", query_current), ("Color histogram baseline", query_color)]:
            scores=[]
            for item in gallery_index:
                ref = safe_open_image(item["path"])
                if ref is None: continue
                rv = np.asarray(item["vector"], dtype=np.float32) if method == "Current feature extractor" else color_descriptor(ref)
                if rv.size == qv.size:
                    score=float(np.dot(qv,rv)/(np.linalg.norm(qv)*np.linalg.norm(rv)+1e-12)); scores.append((score,item["class_name"],item["filename"]))
            if scores:
                score, cls, fn=max(scores)
                result_rows.append({"method":method,"best_class":cls,"best_reference":fn,"best_similarity":score,"gallery_comparable_images":len(scores)})
        if result_rows:
            st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)
            st.caption("Scores from different feature spaces should not be interpreted as directly calibrated or necessarily comparable. Compare class outcomes and evaluate both methods on the same labelled test set for a fair benchmark.")
            st.download_button("Download model comparison", pd.DataFrame(result_rows).to_csv(index=False).encode(), "model_comparison.csv", "text/csv")
    elif upload:
        st.warning("Add gallery images first.")

elif page == "Gallery Quality Checker":
    page_header("Gallery Quality Checker", "Find exact duplicates and flag potentially blurry or low-detail reference images.")
    if not gallery_index:
        st.warning("The gallery is empty.")
    else:
        quality_rows=[]; hashes={}
        for item in gallery_index:
            img=safe_open_image(item["path"])
            if img is None: continue
            gray=np.asarray(img.convert("L").resize((256,256)), dtype=np.float32)
            lap=-4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
            blur_score=float(lap.var())
            digest=hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()
            duplicate_of=hashes.get(digest, "")
            if digest not in hashes: hashes[digest]=item["path"]
            quality_rows.append({"class_name":item["class_name"],"filename":item["filename"],"width":img.width,"height":img.height,"laplacian_variance":round(blur_score,3),"possible_blur":blur_score<80,"exact_duplicate_of":duplicate_of,"path":item["path"]})
        qdf=pd.DataFrame(quality_rows)
        st.metric("Images checked",len(qdf)); st.metric("Exact duplicates",int((qdf["exact_duplicate_of"]!="").sum())); st.metric("Potentially blurry",int(qdf["possible_blur"].sum()))
        st.caption("Blur threshold is a heuristic and may need tuning for your image type. Duplicate detection checks exact decoded pixel content, not near-duplicates.")
        st.dataframe(qdf, use_container_width=True, hide_index=True)
        st.download_button("Download gallery quality report",qdf.to_csv(index=False).encode(),"gallery_quality_report.csv","text/csv")

elif page == "Live Camera Recognition":
    page_header(
        "Live Camera Recognition",
        "Capture an image with your browser camera and compare it with the reference gallery.",
    )
    st.info(
        "Your browser may ask for camera permission. "
        "This captures a still image; it is not continuous video tracking."
    )
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None and gallery_index:
        try:
            camera_image = Image.open(camera_file).convert("RGB")
            st.image(camera_image, caption="Captured image", width=360)
            preds = classify_image(camera_image, gallery_index)
            if preds:
                best = preds[0]
                best_class = best.get("class_name", "Unknown")
                best_score = float(best.get("similarity", 0.0))
                st.success(
                    f"Closest match: {best_class} · similarity {best_score:.4f}"
                )
                st.subheader("Top 5 Predictions")
                for i, pred in enumerate(preds[:5], 1):
                    class_name = pred.get("class_name", "Unknown")
                    similarity = float(pred.get("similarity", 0.0))
                    filename = (
                        pred.get("filename")
                        or pred.get("best_gallery_file")
                        or "Reference filename unavailable"
                    )
                    st.write(
                        f"{i}. **{class_name}** — "
                        f"{similarity:.4f} · {filename}"
                    )
                st.caption(
                    "Similarity is a feature-space score, not a probability. "
                    "A high score does not guarantee a correct prediction."
                )
            else:
                st.warning("No matching gallery images could be compared.")
        except Exception as exc:
            st.error(f"Recognition failed: {exc}")
    elif camera_file is not None:
        st.warning("Add gallery images first.")


elif page == "Model Information":
    page_header("Model Information", "Inspect the model setup, checkpoint, data paths, and limitations.")
    left, right = st.columns(2)
    with left:
        st.subheader("Model status")
        st.write(f"**Load status:** {model_status}")
        st.write(f"**Device:** {device}")
        st.write(f"**Checkpoint exists:** {MODEL_PATH.exists()}")
        if MODEL_PATH.exists():
            st.write(f"**Checkpoint size:** {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB")
            st.write(f"**Last modified:** {datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(timespec='seconds')}")
        if model_message:
            st.warning(model_message)
        if model is not None:
            st.write(f"**Model class:** `{model.__class__.__name__}`")
            st.write(f"**Feature method available:** {hasattr(model, 'extract_features')}")
    with right:
        st.subheader("Project paths")
        st.code(
            f"data/gallery/\ndata/test/\ndata/unlabeled/\nmodels/zerolabel_model.pth\nresults/search_history.csv",
            language="text",
        )
        st.subheader("How recognition works")
        st.markdown(
            "1. Resize and normalize the uploaded image.\n"
            "2. Extract a feature vector with `model.py` when available.\n"
            "3. Compare the vector with gallery image vectors using cosine similarity.\n"
            "4. Rank classes using the average of their strongest gallery matches."
        )
    st.subheader("Limitations and responsible use")
    st.markdown(
        "- A similarity score is not a probability or a guarantee of correctness.\n"
        "- Results depend on gallery coverage, image quality, and the checkpoint's training.\n"
        "- Evaluation is meaningful only when test images are independent and correctly sorted into class folders.\n"
        "- If the model checkpoint is missing, this app may use a simple color-histogram fallback; that fallback is not a trained semantic recognition model."
    )
    st.subheader("Runtime versions")
    st.write(f"Python/PyTorch device: `{device}`")
    st.write(f"PyTorch version: `{torch.__version__}`")
    st.write(f"NumPy version: `{np.__version__}`")
    st.write(f"Pandas version: `{pd.__version__}`")

st.divider()
st.caption("ZeroLabel Vision · Local computer-vision workspace · Results depend on model quality and dataset coverage.")