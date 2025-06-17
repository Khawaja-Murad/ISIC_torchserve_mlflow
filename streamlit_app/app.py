# streamlit_app/app.py
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
except:
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
                    client.transition_model_version_stage(
                        name="skin_vit", version=version, stage="Production"
                    )
                    st.success("Model staged to production!")
                    time.sleep(1)
                    st.rerun()

            with col2:
                if st.button("Stage to Staging"):
                    client.transition_model_version_stage(
                        name="skin_vit", version=version, stage="Staging"
                    )
                    st.success("Model staged to staging!")
                    time.sleep(1)
                    st.rerun()
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

            # Log prediction to MLflow
            try:
                with mlflow.start_run(run_name="Streamlit_Prediction"):
                    # Log prediction metadata
                    mlflow.log_metric("latency_ms", latency * 1000)
                    mlflow.log_param("model_version", current_version)

                    # Log image
                    mlflow.log_image(image, "input_image.jpg")

                    # Handle successful prediction
                    if (
                        len(prediction) > 0
                    ):  # isinstance(prediction, list) and len(prediction) > 0:
                        result = prediction

                        # Display results
                        st.subheader("Analysis Results")
                        st.markdown(
                            f"""
                        **Predicted Condition**: {result.get('class', 'N/A')}  
                        **Confidence**: {result.get('confidence', 0):.2%}  
                        **Class Index**: {result.get('label_index', 'N/A')}
                        """
                        )

                        # Log prediction results
                        mlflow.log_metric("confidence", result.get("confidence", 0))
                        mlflow.log_param(
                            "predicted_class", result.get("class", "unknown")
                        )

                        # Create visualization
                        classes = [
                            "actinic keratosis",
                            "basal cell carcinoma",
                            "dermatofibroma",
                            "melanoma",
                            "nevus",
                            "pigmented benign keratosis",
                            "seborrheic keratosis",
                            "squamous cell carcinoma",
                            "vascular lesion",
                        ]
                        confidences = {c: 0 for c in classes}
                        if "class" in result and "confidence" in result:
                            confidences[result["class"]] = result["confidence"]

                        fig, ax = plt.subplots(figsize=(8, 6))
                        y_pos = np.arange(len(classes))
                        ax.barh(
                            y_pos,
                            [confidences[c] for c in classes],
                            align="center",
                            color="skyblue",
                        )
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(classes)
                        ax.invert_yaxis()
                        ax.set_xlabel("Confidence")
                        ax.set_title("Prediction Probabilities")
                        plt.tight_layout()

                        st.pyplot(fig)

                        # Save and log visualization
                        fig.savefig("prediction_plot.png")
                        mlflow.log_artifact("prediction_plot.png")

                        st.success("✅ Prediction logged to MLflow!")
                    else:
                        st.error(
                            f"❌ Prediction failed: {prediction.get('error', 'Unknown error')}"
                        )
                        mlflow.log_param("error", str(prediction))
            except Exception as e:
                st.error(f"❌ Failed to log to MLflow: {str(e)}")

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
