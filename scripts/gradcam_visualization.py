import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load trained model
model = tf.keras.models.load_model("../breast_cancer_resnet50_cnn_finetuned.keras")

# Ask user for image path
img_path = input("Enter the full path to the test image: ")

# Check if the file exists
if not os.path.isfile(img_path):
    print("❌ File not found. Please check the path.")
    exit()

# Load the image
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Prediction
preds = model.predict(img_array)
pred_class = np.argmax(preds[0])
print(f"Predicted class: {pred_class}, Confidence: {preds[0][pred_class]:.4f}")

# Grad-CAM
last_conv_layer_name = "conv5_block3_out"  # Replace with your last conv layer
grad_model = tf.keras.models.Model([model.inputs],
                                   [model.get_layer(last_conv_layer_name).output, model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:, pred_class]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()[0]
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# Overlay heatmap
heatmap = cv2.resize(heatmap, (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
