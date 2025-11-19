🍽️ Food Image Classification Using Deep Learning (CNN, VGG16, ResNet50)

A complete end-to-end Deep Learning project that classifies food images into predefined categories. This system enables automated food recognition for restaurant menu digitization, calorie estimation, health tracking, and diet monitoring apps.

🧠 Project Overview

Food image classification is a challenging computer vision task due to variations in lighting, presentation styles, and similarity among dishes.
This project uses multiple deep learning models — Custom CNN, VGG16, and ResNet50 — to classify food images with high accuracy.

The pipeline automates:

✔ Dataset splitting
✔ Nutrition metadata generation
✔ Dynamic model evaluation (Precision, Recall, F1-score)
✔ Performance JSON creation
✔ Multiple model benchmarking

🚀 Goal

Classify food images into their respective classes

📈 Best Model: VGG16

🎯 Best F1-Score: Varies depending on dataset

🧩 Architecture
graph TD
A[Dataset Loading] --> B[Class Extraction]
B --> C[JSON Nutrition Generation]
C --> D[Data Splitting - Train/Val/Test]
D --> E[Model Detection - Input Shape Extraction]
E --> F[Prediction & Evaluation]
F --> G[Performance Metrics JSON]
G --> H[Model Comparison & Reporting]

🧰 Tech Stack & Libraries
Category	Tools / Libraries
Language	Python 3
Deep Learning	TensorFlow / Keras
Models Used	CNN, VGG16, ResNet50
ML Metrics	Accuracy, Precision, Recall, F1-Score
Visualization	Matplotlib
Utilities	NumPy, Pandas, JSON
Deployment Ready	Flask, Gunicorn
🗃 Dataset Description

Source:
https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset

Folder structure:
Food Classification dataset/
├── Baked potato/
├── samosa/
├── Taco/
├── pizza/
└── ... more classes


Total Classes: Dynamically detected

Each image belongs to exactly one food category

📊 Data Preprocessing Steps

Automatic class detection

Train/Validation/Test Split

70% Training

15% Validation

15% Testing

Image Rescaling: 1/255

Dynamic resizing based on model input shape

Model predictions on test set

🧮 Models Used

You trained and evaluated multiple .h5 models automatically:

✔ Custom CNN
✔ VGG16 (Pre-trained on ImageNet)
✔ ResNet50

For each model, the following are saved:

Input shape

Precision

Recall

F1-score

Performance JSON file

🧾 Model Evaluation
Metrics stored for every model:
Metric	Description
Precision	Macro averaged precision
Recall	Macro averaged recall
F1-Score	Macro F1-score
Input Shape	Dynamic model input
Example JSON output:
{
  "input_shape": [224, 224, 3],
  "precision": 0.89,
  "recall": 0.87,
  "f1_score": 0.88
}

💻 Core Functionalities (from main.py)
1️⃣ Class Extraction

Scans dataset folders and identifies food classes.

2️⃣ Nutrition JSON Creation

Generates random nutritional values for each food item.

3️⃣ Dataset Splitting

Creates the following structure:

food_data_splitting/
├── training_data/
├── validation_data/
└── testing_data/

4️⃣ Model JSON Generation

Detects input shapes of all .h5 models in:

Trained_models/

5️⃣ Performance Evaluation

For each model, generates:

Precision

Recall

F1-score

Confusion matrix

Saved in:

model_performance/

🧩 Project Structure
├── main.py
├── app.py (optional for UI)
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

📈 Visual Insights

During evaluation:

🔹 VGG16 performs the best
🔹 ResNet50 delivers stable results
🔹 Custom CNN performs well on simpler classes
🔹 Larger input sizes increase accuracy but require more memory
🔹 F1-score is the best comparison metric for imbalanced data

⚙ Installation & Usage
Clone the repository
git clone https://github.com/username/food-image-classification.git
cd food-image-classification

Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# OR
source venv/bin/activate  # Mac/Linux

Install dependencies
pip install -r requirements.txt

Run the main pipeline
python main.py

🧑‍💻 Author

👨‍🎓 P. Naveen Kumar
📧 Email: puppalanaveenkumar11@gmail.com

🌟 Acknowledgements

TensorFlow & Keras

Scikit-learn

Matplotlib

Vihara Tech (Guidance)

Open-source Deep Learning Community

🧭 Future Enhancements

. Real-time Food Detection: Live camera food recognition

. Calorie & Nutrition Estimation: Automatic calorie prediction

. Multi-label Classification: Detect multiple items in a single image

. Mobile & Edge Deployment: Convert to TensorFlow Lite / ONNX

. User Personalization: Food tracking and analytics

. Dataset Expansion: Add more global cuisine categories

. Web UI: Build a fully interactive Flask interface
