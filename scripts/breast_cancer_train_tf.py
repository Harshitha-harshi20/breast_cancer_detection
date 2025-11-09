import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import os

# -----------------------------
# 1. Paths
# -----------------------------
dataset_dir = r"C:\Users\sonuh\breast_cancer_detection\dataset"
train_csv = os.path.join(dataset_dir, "split_dataset", "train_labels.csv")
val_csv   = os.path.join(dataset_dir, "split_dataset", "val_labels.csv")
test_csv  = os.path.join(dataset_dir, "split_dataset", "test_labels.csv")

# -----------------------------
# 2. Load CSVs
# -----------------------------
train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
test_df  = pd.read_csv(test_csv)

# -----------------------------
# 3. Make sure full paths are used
# -----------------------------
# Replace 'filename' with your actual column name if different
if 'filename' in train_df.columns:
    train_df['filename'] = train_df['filename'].apply(lambda x: os.path.join(dataset_dir, "images", x))
    val_df['filename'] = val_df['filename'].apply(lambda x: os.path.join(dataset_dir, "images", x))
    test_df['filename'] = test_df['filename'].apply(lambda x: os.path.join(dataset_dir, "images", x))
    img_col = 'filename'
elif 'image_path' in train_df.columns:
    img_col = 'image_path'
else:
    raise ValueError("Cannot find column with image paths in CSV")

label_col = 'label'  # Change if your label column has a different name

# -----------------------------
# 4. Data Generators
# -----------------------------
image_size = (224, 224)
batch_size = 32
num_classes = 3

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.9,1.1],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col=img_col,
    y_col=label_col,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col=img_col,
    y_col=label_col,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col=img_col,
    y_col=label_col,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# -----------------------------
# 5. Model
# -----------------------------
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Freeze base initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 6. Callbacks
# -----------------------------
checkpoint = ModelCheckpoint('breast_cancer_model_tf.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

# -----------------------------
# 7. Train
# -----------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[checkpoint, early_stop]
)

# -----------------------------
# 8. Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
