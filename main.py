import numpy as np
import pandas as pd
import sys
import sklearn
import os
import random
import json
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score, precision_score, recall_score,f1_score
import matplotlib.pyplot as plt
from log import setup_logging
logger = setup_logging('main')

import warnings
warnings.filterwarnings('ignore')


class food_classification:
    def __init__(self, path):
        self.path = path

    def finding_classes(self):
        self.class_names = []
        for class_name in os.listdir(self.path):
            if os.path.isdir(os.path.join(self.path, class_name)):
                self.class_names.append(class_name)
        logger.info(f'Dataset loaded: {self.class_names}')
        logger.info(f'Number of classes: {len(self.class_names)}')

    def jsonfile_creation(self):
        try:
            self.nutrition_info = []
            for food_name in self.class_names:
                food_nutrition = {
                    "food_name": food_name,
                    "nutritional_info": {
                        "protein": f"{random.randint(1, 30)}",
                        "fiber": f"{random.randint(1, 10)}",
                        "calories": random.randint(100, 600),
                        "carbohydrates": f"{random.randint(10, 70)}",
                        "fat": f"{random.randint(1, 25)}"
                    }
                }
                self.nutrition_info.append(food_nutrition)

            json_path = 'D:\\Food classification project\\food_nutrition.json'
            with open(json_path, 'w') as json_file:
                json.dump(self.nutrition_info, json_file, indent=4)

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def data_splitting_for_training(self):
        try:
            output_folder = "food_data_splitting"
            image_per_class = 300

            train_ratio = 0.7
            val_ratio = 0.15
            test_ratio = 0.15

            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)

            for split in ['training_data', 'validation_data', 'testing_data']:
                os.makedirs(os.path.join(output_folder, split), exist_ok=True)

            for class_name in os.listdir(self.path):
                class_path = os.path.join(self.path, class_name)
                if not os.path.isdir(class_path):
                    continue

                images = [i for i in os.listdir(class_path) if i.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(images) == 0:
                    continue

                sampled_images = random.sample(images, image_per_class) if len(images) >= image_per_class else images

                train_imgs, temp_imgs = train_test_split(sampled_images, test_size=(1 - train_ratio), random_state=42)
                val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio / (val_ratio + test_ratio),
                                                       random_state=42)
                logger.info(f"{class_name}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")
                for split, img_list in zip(['training_data', 'validation_data', 'testing_data'],
                                           [train_imgs, val_imgs, test_imgs]):
                    split_dir = os.path.join(output_folder, split, class_name)
                    os.makedirs(split_dir, exist_ok=True)

                    for img_name in img_list:
                        shutil.copy(os.path.join(class_path, img_name), os.path.join(split_dir, img_name))

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def json_files_of_models(self):
        try:
            model_path = 'D:\\Food classification project\\Trained_models'
            models_json = {
                "all_models": []
            }

            for file in os.listdir(model_path):
                if file.endswith(".h5"):
                    full_path = os.path.join(model_path, file)
                    model = load_model(full_path, compile=False)

                    input_shape = tuple(model.input_shape[1:])  # (height, width, channels)

                    models_json["all_models"].append({
                        "name": file,
                        "input_shape": input_shape
                    })

            json_path = 'D:\\Food classification project\\All_models.json'
            with open(json_path, "w") as f:
                json.dump(models_json, f, indent=4)

            print("Model JSON file created successfully with input shapes.")

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def model_performance(self):
        """Use each model's real input shape"""
        try:
            json_path = 'D:\\Food classification project\\All_models.json'
            with open(json_path, "r") as f:
                models_data = json.load(f)

            test_dir = "food_data_splitting/testing_data"
            model_folder = "D:\\Food classification project\\Trained_models"
            performance_dir = "D:\\Food classification project\\model_performance"
            os.makedirs(performance_dir, exist_ok=True)

            for model_info in models_data["all_models"]:
                model_file = model_info["name"]
                input_shape = tuple(model_info["input_shape"])  # (h, w, c)

                datagen = ImageDataGenerator(rescale=1.0 / 255)
                test_set = datagen.flow_from_directory(
                    test_dir,
                    target_size=input_shape[:2],  # dynamic input size
                    batch_size=20,
                    class_mode='categorical',
                    shuffle=False
                )

                model_path = os.path.join(model_folder, model_file)
                model = load_model(model_path, compile=False)

                predictions = model.predict(test_set)
                predicted_classes = np.argmax(predictions, axis=1)
                true_classes = test_set.classes

                acc = accuracy_score(true_classes, predicted_classes)
                precision = precision_score(true_classes, predicted_classes, average="macro", zero_division=0)
                recall = recall_score(true_classes, predicted_classes, average="macro", zero_division=0)
                f1_score_of_pred= f1_score(true_classes, predicted_classes, average="macro", zero_division=0)
                cm = confusion_matrix(true_classes, predicted_classes)

                json_output = {
                    "input_shape": input_shape,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score":float(f1_score_of_pred)
                }

                json_file_path = os.path.join(
                    performance_dir,
                    model_file.replace(".h5", "_performance.json")
                )

                with open(json_file_path, "w") as jf:
                    json.dump(json_output, jf, indent=4)

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.error(f"Performance Error : {er_lin.tb_lineno} : due to {er_msg}")


if __name__ == "__main__":
    path = 'D:\\Food classification project\\Food Classification dataset'
    obj = food_classification(path)
    obj.finding_classes()
    obj.jsonfile_creation()
    obj.data_splitting_for_training()
    obj.json_files_of_models()
    obj.model_performance()
