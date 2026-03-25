"""
Grad-CAM explainability for skin cancer predictions.
Required for every prediction output — clinical interpretability standard.
"""

import numpy as np
import cv2
import tensorflow as tf
from typing import Tuple


CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
LAST_CONV_LAYER = {
    'efficientnetb4': 'top_activation',
    'resnet50':       'conv5_block3_out',
    'densenet121':    'relu',
}


def generate_gradcam(model: tf.keras.Model,
                     image: np.ndarray,
                     class_idx: int,
                     last_conv_layer_name: str = 'top_activation') -> np.ndarray:
    """
    Generate Grad-CAM heatmap for the predicted class.

    Args:
        model:               Trained Keras model
        image:               Preprocessed image array (224, 224, 3), values in [0,1]
        class_idx:           Index of the target class
        last_conv_layer_name: Name of the last conv layer to hook

    Returns:
        heatmap (np.ndarray): Normalized heatmap of shape (224, 224)
    """
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(image[np.newaxis, ...], tf.float32)
        conv_output, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(image: np.ndarray,
                    heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.

    Args:
        image:   Original image (224, 224, 3), uint8 [0,255]
        heatmap: Grad-CAM heatmap (H, W), float [0,1]
        alpha:   Heatmap blend weight

    Returns:
        Superimposed image (224, 224, 3), uint8
    """
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed


def predict_with_explanation(model: tf.keras.Model,
                             image: np.ndarray,
                             model_name: str = 'efficientnetb4'
                             ) -> Tuple[str, float, np.ndarray, dict]:
    """
    Full prediction pipeline with Grad-CAM explanation.

    Returns:
        (predicted_class, confidence, gradcam_overlay, class_probabilities)
    """
    img_tensor = image[np.newaxis, ...].astype(np.float32)
    probs = model.predict(img_tensor, verbose=0)[0]
    class_idx = np.argmax(probs)
    confidence = float(probs[class_idx])
    predicted_class = CLASS_NAMES[class_idx]

    last_conv = LAST_CONV_LAYER.get(model_name, 'top_activation')
    heatmap = generate_gradcam(model, image, class_idx, last_conv)

    # Denormalize image for overlay (assumes ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_display = np.clip((image * std + mean) * 255, 0, 255).astype(np.uint8)
    gradcam_overlay = overlay_gradcam(img_display, heatmap)

    class_probs = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}

    return predicted_class, confidence, gradcam_overlay, class_probs
