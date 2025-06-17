import streamlit as st
import requests
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import time

# Configuration
MLFLOW_PORT = 5001
st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="wide")

# MLflow setup
mlflow.set_tracking_uri(f"http://localhost:{MLFLOW_PORT}")


# Prediction function
def predict_image(image_bytes):
    try:
        response = requests.post(
            "http://localhost:8080/predictions/skin_vit",
            data=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=10,
        )
        return (
            response.json()
            if response.status_code == 200
            else {"error": f"HTTP {response.status_code}"}
        )
    except Exception as e:
        return {"error": str(e)}


def plot_prediction(image, prediction_result):
    """Create visualization with all class probabilities"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Display image
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis("off")

    # Get class names and confidences
    classes = list(prediction_result.keys())
    confidences = [prediction_result[cls] for cls in classes]

    # Sort by confidence
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_confidences = [confidences[i] for i in sorted_indices]

    # Create colormap - highest confidence in green
    colors = [
        "#2ecc71" if i == sorted_indices[0] else "#3498db" for i in range(len(classes))
    ]

    # Display prediction probabilities
    y_pos = np.arange(len(classes))
    bars = ax2.barh(y_pos, sorted_confidences, align="center", color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_classes)
    ax2.invert_yaxis()
    ax2.set_xlabel("Confidence")
    ax2.set_title("Prediction Probabilities")
    ax2.set_xlim(0, 1)

    # Add confidence values to bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            ha="left",
            va="center",
        )

    # Highlight top prediction
    ax2.text(
        0.95,
        0.5,
        f"TOP PREDICTION: {sorted_classes[0]}",
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="#2ecc71", alpha=0.3),
    )

    plt.tight_layout()
    return fig


# Main application
st.title("🩺 Skin Lesion Classification")
st.markdown(
    "Upload an image of a skin lesion to classify it using our deep learning model."
)

# Model Registry Info
try:
    client = mlflow.MlflowClient()
    model_versions = client.search_model_versions("name='skin_vit'")
    current_version = (
        max([int(mv.version) for mv in model_versions]) if model_versions else 1
    )
except Exception as e:
    st.warning(f"Couldn't connect to MLflow: {str(e)}")
    model_versions = []
    current_version = 1

# Sidebar - Model Management
with st.sidebar:
    st.header("Model Registry")
    st.metric("Current Model Version", current_version)

    if model_versions:
        version = st.selectbox("Select Version", [mv.version for mv in model_versions])
        mv = next((m for m in model_versions if m.version == version), None)

        if mv:
            st.write(f"**Status:** {mv.status}")
            st.write(f"**Registered:** {mv.creation_timestamp}")
            st.write(f"**Description:** {mv.description or 'None'}")

            # Model stage management
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Stage to Production"):
                    try:
                        client.transition_model_version_stage(
                            name="skin_vit", version=version, stage="Production"
                        )
                        st.success("Model staged to production!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to stage model: {str(e)}")

            with col2:
                if st.button("Stage to Staging"):
                    try:
                        client.transition_model_version_stage(
                            name="skin_vit", version=version, stage="Staging"
                        )
                        st.success("Model staged to staging!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to stage model: {str(e)}")
    else:
        st.warning("No models in registry")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a skin lesion image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Convert to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    # Prediction button
    if st.button("Analyze Image"):
        with st.spinner("Analyzing lesion..."):
            start_time = time.time()
            prediction = predict_image(img_bytes)
            latency = time.time() - start_time

            # Handle prediction response
            if (
                len(prediction) > 0
            ):  # isinstance(prediction, list) and len(prediction) > 0:
                result = prediction  # [0]

                # Display results
                st.subheader("Analysis Results")
                st.markdown(
                    f"""
                **Predicted Condition**: {result.get('class', 'N/A')}  
                **Confidence**: {result.get('confidence', 0):.2%}  
                **Class Index**: {result.get('label_index', 'N/A')}
                """
                )

                # Create visualization if we have all confidences
                if "all_confidences" in result:
                    fig = plot_prediction(image, result["all_confidences"])
                    st.pyplot(fig)

                    # Log to MLflow
                    try:
                        with mlflow.start_run(run_name="Streamlit_Prediction"):
                            # Log basic info
                            mlflow.log_metric("latency_ms", latency * 1000)
                            mlflow.log_param("model_version", current_version)
                            mlflow.log_metric("confidence", result.get("confidence", 0))
                            mlflow.log_param(
                                "predicted_class", result.get("class", "unknown")
                            )

                            # Log image
                            mlflow.log_image(image, "input_image.jpg")

                            # Log all probabilities
                            for cls, conf in result["all_confidences"].items():
                                mlflow.log_metric(f"prob_{cls}", conf)

                            # Save and log visualization
                            fig.savefig("prediction_plot.png")
                            mlflow.log_artifact("prediction_plot.png")

                            st.success("✅ Prediction logged to MLflow!")
                    except Exception as e:
                        st.error(f"❌ MLflow logging failed: {str(e)}")
                else:
                    st.warning("⚠️ Full confidence data not available")
            else:
                st.error(
                    f"❌ Prediction failed: {prediction.get('error', 'Unknown error')}"
                )

# Model Performance Section
st.header("Model Performance Monitoring")
st.markdown("### Recent Prediction Metrics")

try:
    # Get last 10 predictions from MLflow
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=10)

    if not runs.empty:
        # Filter successful predictions
        successful_runs = runs[runs["metrics.confidence"].notnull()]

        if not successful_runs.empty:
            # Display metrics
            avg_confidence = successful_runs["metrics.confidence"].mean()
            avg_latency = successful_runs["metrics.latency_ms"].mean()

            col1, col2 = st.columns(2)
            col1.metric("Average Confidence", f"{avg_confidence:.2%}")
            col2.metric("Average Latency", f"{avg_latency:.1f} ms")

            # Confidence distribution
            st.bar_chart(successful_runs[["metrics.confidence"]])

            # Latency over time
            st.line_chart(runs.set_index("start_time")[["metrics.latency_ms"]])
        else:
            st.warning("No successful predictions recorded yet")
    else:
        st.info("No prediction data available")
except Exception as e:
    st.error(f"Couldn't load metrics: {str(e)}")

# Link to MLflow UI
st.divider()
st.markdown(f"[Open MLflow UI to explore full metrics](http://localhost:{MLFLOW_PORT})")
