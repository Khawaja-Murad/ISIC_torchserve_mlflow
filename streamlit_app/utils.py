# streamlit_app/utils.py
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io


def predict_image(image_bytes):
    """Send image to TorchServe for prediction"""
    response = requests.post(
        "http://localhost:8080/predictions/skin_vit",
        data=image_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    return response.json()


def plot_prediction(image, prediction_result):
    """Create visualization of prediction results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Display image
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis("off")

    # Display prediction probabilities
    classes = list(prediction_result.keys())
    confidences = list(prediction_result.values())
    y_pos = np.arange(len(classes))

    ax2.barh(y_pos, confidences, align="center", color="skyblue")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_xlabel("Confidence")
    ax2.set_title("Prediction Probabilities")

    return fig
