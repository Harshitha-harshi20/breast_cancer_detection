import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Paths ==========
DATASET_DIR = "dataset/split_dataset"
MODEL_PATH = "breast_cancer_resnet50_cnn_finetuned.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ========== Load Model ==========
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ========== Load Test Dataset ==========
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = test_ds.class_names
print(f"📂 Classes: {class_names}")

# ========== Predict ==========
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# ========== Confusion Matrix ==========
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ========== Visualization ==========
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Breast Cancer Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ========== Classification Report ==========
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))
