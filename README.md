🍽️ Food Image Classification Using Deep Learning

CNN | VGG16 | ResNet50 | Flask Deployment

A complete end-to-end Food Image Recognition System built using Deep Learning.
This project classifies food images into labeled categories using multiple models, auto-generates nutrition data, evaluates models with performance metrics, and prepares the system for deployment.

🧠 Project Overview

Food recognition from images is a challenging task due to variations in lighting, angle, presentation, and similarity across food categories.
This system automates the entire process—dataset management, model evaluation, metric generation, and deployment-ready outputs.

The pipeline includes:

✔ Automated dataset splitting
✔ Nutrition metadata generation
✔ Multiple model evaluation (CNN, VGG16, ResNet50)
✔ Precision, Recall, F1-score, Confusion Matrix
✔ Best-model detection
✔ Model performance JSON logs
✔ Flask-ready prediction API

🚀 Goal

Classify food images into predefined categories with high accuracy and support downstream tasks like calorie estimation and restaurant digitization.

🏆 Best Performing Model

VGG16 delivered the highest F1-score in most experiments.

🧩 System Architecture
graph TD
A[📁 Dataset Loading] --> B[🔍 Class Extraction]
B --> C[🥗 Nutrition JSON Generation]
C --> D[🔀 Train/Val/Test Split]
D --> E[🧠 Model Input Shape Detection]
E --> F[🎯 Prediction & Evaluation]
F --> G[📊 Performance Metrics JSON]
G --> H[🏆 Best Model Comparison & Reporting]

🧰 Tech Stack
Category	Tools / Libraries
Language	Python 3
Framework	TensorFlow / Keras
Deep Learning Models	Custom CNN, VGG16, ResNet50
Metrics	Accuracy, Precision, Recall, F1-Score
Deployment	Flask, Gunicorn
Utilities	NumPy, Pandas, JSON
Visualization	Matplotlib
🗃 Dataset

Source:
https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset

Folder structure:

Food Classification dataset/
├── Baked potato/
├── samosa/
├── Taco/
├── pizza/
└── ... more classes


✔ Total classes: Dynamically detected
✔ Each class = one food category
✔ Dataset automatically split into train/val/test

📊 Data Preprocessing

The pipeline automatically performs:

Class extraction

Train/Val/Test split (70/15/15)

Rescaling (1/255)

Adaptive image resizing based on model input shape

Nutrition JSON creation for each class

Test set predictions

Confusion matrix generation

🧠 Deep Learning Models

You trained and evaluated multiple .h5 models:

✔ Custom CNN

Lightweight model for faster inference

✔ VGG16

Best performance in most evaluations

✔ ResNet50

Stable and deep model for complex class boundaries

Each model logs:

Input shape

Precision

Recall

F1-score

Confusion matrix

JSON performance file

Example:

{
  "input_shape": [224, 224, 3],
  "precision": 0.89,
  "recall": 0.87,
  "f1_score": 0.88
}

🧩 Project Structure
├── main.py
├── app.py                  # Flask API (optional)
├── Food Classification dataset/
├── food_data_splitting/
│   ├── training_data/
│   ├── validation_data/
│   └── testing_data/
├── Trained_models/
├── model_performance/
├── All_models.json
├── food_nutrition.json
├── requirements.txt
└── README.md

📈 Evaluation Results

🔹 VGG16 – Best F1 Score
🔹 ResNet50 – Strong and consistent
🔹 Custom CNN – Lightweight, fast, good for simple datasets

Observations:

Larger input image sizes → Higher accuracy

F1-score provides the best comparison for imbalanced classes

Pretrained models outperform custom CNN

⚙ Installation & Usage
1️⃣ Clone the repository
git clone https://github.com/username/food-image-classification.git
cd food-image-classification

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# OR
source venv/bin/activate   # macOS/Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the full pipeline
python main.py

Optional: Run Flask app
python app.py

🧑‍🎓 Author

P. Naveen Kumar
📧 Email: puppalanaveenkumar11@gmail.com

🌟 Acknowledgements

TensorFlow & Keras

Scikit-learn

Matplotlib

Vihara Tech (Guidance)

Open-source AI Community

🧭 Future Enhancements

📸 Real-time food detection (camera input)

🔢 Calorie & nutrition estimation

🍱 Multi-label classification for meals

📱 Mobile deployment using TensorFlow Lite / ONNX

👤 User-based food tracking analytics

🍜 Dataset expansion with global cuisines

🌐 Complete web UI for interactive predictions

If you'd like, I can also:
✅ Generate a project logo/banner
✅ Create a requirements.txt
✅ Build a GitHub Pages portfolio site
✅ Add badges (build, license, stars, datasets)
