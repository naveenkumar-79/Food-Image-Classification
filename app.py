# app.py
import os
import sys
import json
import shutil
import random
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

try:
    from log import setup_logging
    logger = setup_logging('main')
except Exception:
    # fallback basic logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("main")

# ---------- CONFIG ----------
BASE_DIR = Path("D:/Food classification project")  # set to your project base
DATASET_DIR = BASE_DIR / "Food Classification dataset"  # where the original dataset classes are stored
TRAINED_MODELS_DIR = BASE_DIR / "Trained_models"
MODEL_PERF_DIR = BASE_DIR / "model_performance"
NUTRITION_JSON = BASE_DIR / "food_nutrition.json"

UPLOAD_FOLDER = Path("static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB limit

# The explicit classes you listed (will also be derived from dataset folder if present)
HARDCODED_CLASSES = [
    'Baked Potato', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Sandwich', 'Taco', 'Taquito',
    'apple_pie', 'burger', 'butter_naan', 'chai', 'chapati', 'cheesecake', 'chicken_curry',
    'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'ice_cream', 'idli', 'jalebi',
    'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 'omelette', 'paani_puri',
    'pakode', 'pav_bhaji', 'pizza', 'samosa', 'sushi'
]

# Map model family -> folder/file prefix used in your Trained_models folder
FAMILY_PREFIX = {
    "cnn": "cnn_",
    "vgg16": "vgg16",
    "resnet50": "resnet"
}

# ---------- FLASK APP ----------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def load_classes():
    """
    Prefer dataset folder-derived classes if available, otherwise fall back to HARDCODED_CLASSES.
    NOTE: Class order must match model training class order for correct label mapping.
    """
    try:
        classes = []
        if DATASET_DIR.exists():
            for c in sorted(os.listdir(DATASET_DIR)):
                if os.path.isdir(DATASET_DIR / c):
                    classes.append(c)
        if classes:
            logger.info(f"Loaded {len(classes)} classes from dataset folder.")
            return classes
    except Exception as e:
        logger.warning(f"Could not read dataset dir: {e}")

    logger.info("Falling back to hardcoded classes.")
    return HARDCODED_CLASSES


CLASSES = load_classes()


def predict_with_model(model_path, img_path):
    """
    Load model, read its input shape, preprocess image accordingly, predict and return (predicted_index, confidence, model_name).
    """
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        logger.error(f"Failed to load model {model_path}: {e}")
        return None

    # infer input shape from model.input_shape
    try:
        # model.input_shape might be (None, h, w, c) or (None, None, None, 3) etc.
        input_shape = model.input_shape  # tuple
        # take height,width = positions 1,2 if possible
        if len(input_shape) >= 3 and input_shape[1] and input_shape[2]:
            target_size = (int(input_shape[1]), int(input_shape[2]))
        else:
            # fallback target size
            target_size = (224, 224)
            logger.debug(f"Model {model_path} has unusual input_shape {input_shape}; using fallback {target_size}")
    except Exception as e:
        logger.warning(f"Failed to determine input shape for {model_path}: {e}")
        target_size = (224, 224)

    # load and preprocess image
    try:
        img = load_img(img_path, target_size=target_size)
        arr = img_to_array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)  # batch dim
    except Exception as e:
        logger.error(f"Failed to load/preprocess image {img_path} for model {model_path}: {e}")
        return None

    try:
        preds = model.predict(arr)
        if preds.ndim == 2:
            probs = preds[0]
        else:
            probs = np.squeeze(preds)

        pred_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        model_name = os.path.basename(model_path)
        # clean up Keras session if necessary (depends on TF version)
        # from tensorflow.keras import backend as K
        # K.clear_session()  # optional, commented out for performance
        return {"model_name": model_name, "pred_idx": pred_idx, "confidence": confidence, "input_shape": target_size}
    except Exception as e:
        logger.error(f"Prediction error for {model_path}: {e}")
        return None


@app.route("/", methods=["GET"])
def index():
    # no prediction shown
    return render_template("index.html", classes=CLASSES, predicted_label=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # file validation
        if "image" not in request.files:
            logger.warning("No file part in request")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            logger.warning("No selected file")
            return redirect(request.url)

        if not allowed_file(file.filename):
            logger.warning("File type not allowed")
            return redirect(request.url)

        # model family selection
        model_family = request.form.get("model_family")
        if model_family not in FAMILY_PREFIX:
            logger.warning("Invalid model_family selected")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_filename = f"{timestamp}_{filename}"
        save_path = UPLOAD_FOLDER / save_filename
        file.save(save_path)
        logger.info(f"Saved upload to {save_path}")

        # discover model folders/files for chosen family
        prefix = FAMILY_PREFIX[model_family]
        candidates = []

        if TRAINED_MODELS_DIR.exists():
            for entry in sorted(TRAINED_MODELS_DIR.iterdir()):
                # If entry is a folder and startswith prefix OR entry is a model file start-with prefix
                if entry.is_dir() and entry.name.lower().startswith(prefix.lower()):
                    # find any .h5 file(s) inside this model folder
                    for sub in entry.rglob("*.h5"):
                        candidates.append(sub)
                elif entry.is_file() and entry.suffix.lower() == ".h5" and entry.name.lower().startswith(prefix.lower()):
                    candidates.append(entry)

        if not candidates:
            logger.warning(f"No models found for family '{model_family}' with prefix '{prefix}' in {TRAINED_MODELS_DIR}")
            return render_template("index.html", classes=CLASSES, predicted_label="No model found", model_used="", confidence="", performance={}, nutrition={})

        # iterate through candidate models and predict
        predictions = []
        for model_path in candidates:
            logger.info(f"Predicting using model: {model_path}")
            res = predict_with_model(str(model_path), str(save_path))
            if res:
                res["model_path"] = str(model_path)
                predictions.append(res)

        if not predictions:
            logger.error("All model predictions failed")
            return render_template("index.html", classes=CLASSES, predicted_label="Prediction failed", model_used="", confidence="", performance={}, nutrition={})

        # choose best prediction by highest confidence
        best = max(predictions, key=lambda x: x["confidence"])
        predicted_index = best["pred_idx"]
        predicted_label = CLASSES[predicted_index] if predicted_index < len(CLASSES) else f"Index_{predicted_index}"
        model_used = best["model_name"]
        confidence = f"{best['confidence']*100:.2f}%"

        # load performance json for this model (if exists)
        # ---------------- CONFUSION MATRIX LOADING -----------------
        perf_json = {}
        expected_perf_fn=MODEL_PERF_DIR/model_used.replace(".h5","_performance.json")
        class_metrics = {"TP": "-", "TN": "-", "FP": "-", "FN": "-"}

        try:
            if expected_perf_fn.exists():
                with open(expected_perf_fn, "r") as pf:
                    perf_json = json.load(pf)
            else:
                logger.info(f"No performance JSON for model {model_used} at {expected_perf_fn}")

            # 1. Load the same model used for prediction
            model_path = os.path.join(str(TRAINED_MODELS_DIR), model_used)
            model = load_model(model_path, compile=False)

            # 2. Get correct input size for test-set evaluation
            input_shape = model.input_shape
            if len(input_shape) >= 3 and input_shape[1] and input_shape[2]:
                target_size = (int(input_shape[1]), int(input_shape[2]))
            else:
                target_size = (224, 224)

            # 3. Load full test dataset (same classes as training)
            test_dir = BASE_DIR / "food_data_splitting/testing_data"
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            datagen = ImageDataGenerator(rescale=1.0 / 255)

            test_set = datagen.flow_from_directory(
                test_dir,
                target_size=target_size,
                batch_size=20,
                class_mode='categorical',
                shuffle=False
            )

            # 4. Predict on entire test dataset
            preds = model.predict(test_set)
            predicted_all = np.argmax(preds, axis=1)
            true_all = test_set.classes

            # 5. Compute confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(true_all, predicted_all)

            # 6. Extract TP/TN/FP/FN for the predicted class only
            cls = predicted_index  # predicted class of uploaded image

            TP = cm[cls][cls]
            FP = np.sum(cm[:, cls]) - TP
            FN = np.sum(cm[cls]) - TP
            TN = np.sum(cm) - (TP + FP + FN)

            class_metrics = {
                "TP": int(TP),
                "FP": int(FP),
                "FN": int(FN),
                "TN": int(TN)
            }

        except Exception as e:
            logger.warning(f"Class metrics error: {e}")

        except Exception as e:
            logger.warning(f"Error reading performance file: {e}")

        # load nutrition info for predicted label
        nutrition = {}
        try:
            if NUTRITION_JSON.exists():
                with open(NUTRITION_JSON, "r") as nf:
                    all_nut = json.load(nf)
                    # all_nut is expected to be list of dicts with "food_name" key
                    for entry in all_nut:
                        if str(entry.get("food_name")).lower() == str(predicted_label).lower():
                            nutrition = entry.get("nutritional_info", {})
                            break
            else:
                logger.info(f"No nutrition json found at {NUTRITION_JSON}")
        except Exception as e:
            logger.warning(f"Error reading nutrition file: {e}")

        # build image path for template (Flask static)
        image_path = str(save_path).replace("\\", "/")  # template uses /{{ image_path }}

        # Render template with prediction
        return render_template(
            "index.html",
            classes=CLASSES,
            predicted_label=predicted_label,
            model_used=model_used,
            confidence=confidence,
            performance=perf_json,
            class_metrics=class_metrics,
            nutrition=nutrition,
            image_path=image_path
        )

    except Exception as e:
        logger.exception(f"Predict endpoint error: {e}")
        return render_template("index.html", classes=CLASSES, predicted_label="Internal error", model_used="", confidence="", performance={}, nutrition={})


if __name__ == "__main__":
    # run dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
