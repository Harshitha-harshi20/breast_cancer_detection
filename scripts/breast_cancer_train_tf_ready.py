import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight

# ======================
# Paths and Parameters
# ======================
DATASET_DIR = "dataset/split_dataset"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE

# ======================
# Load datasets
# ======================
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
print(f"✅ Classes detected: {class_names}")

# ======================
# Optimize pipeline (parallel processing)
# ======================
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ======================
# Data Augmentation
# ======================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# ======================
# Build Model (ResNet50 + CNN head)
# ======================
base_model = keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze base initially

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = keras.applications.resnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

# ======================
# Compile Model
# ======================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================
# Compute Class Weights
# ======================
y_train = np.concatenate([y for x, y in train_ds], axis=0)
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))
print(f"✅ Class weights: {class_weights_dict}")

# ======================
# Callbacks
# ======================
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_resnet50_cnn.keras", save_best_only=True),
    keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# ======================
# Train initial frozen model
# ======================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ======================
# Fine-tune top layers of ResNet50
# ======================
base_model.trainable = True
for layer in base_model.layers[:-50]:  # freeze lower layers
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS//2,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# ======================
# Save final model
# ======================
model.save("breast_cancer_resnet50_cnn_finetuned.keras")
print("🎉 Training complete! Model saved as breast_cancer_resnet50_cnn_finetuned.keras")

# ✅ Evaluate on test set
test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
print(f"\n📊 Final Test Results:")
print(f"   Test Accuracy: {test_acc * 100:.2f}%")
print(f"   Test Loss: {test_loss:.4f}")
