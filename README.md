# 🧠 Skin Lesion Classification System

## A Complete MLOps Pipeline for Medical Image Diagnosis

This project delivers an end-to-end deep learning solution for classifying **skin lesions into 9 diagnostic categories**. It uses a fine-tuned Vision Transformer (ViT) model trained on the ISIC 2019 dataset via **Kaggle**, and integrates seamlessly with **TorchServe** for model serving, **MLflow** for experiment tracking and model versioning, **Streamlit** for an interactive diagnostic UI, and **Docker/Docker Compose** for containerized deployment.

---

## 🔍 Key Highlights

* 🧠 **ViT-based Classifier**: Trained on Kaggle using state-of-the-art techniques
* 🔁 **MLOps-Ready**: Includes model serving, tracking, UI, and containerization — locally deployable
* 📈 **MLflow Integration**: Log, version, and monitor experiments and predictions
* 🔥 **TorchServe API**: Serve the model via REST API using `skin_vit.mar`
* 🌐 **Streamlit Frontend**: Drag & drop interface for quick analysis
* 🐳 **Dockerized**: Fully containerized with Docker Compose for reproducible setups
* 🔒 **Local & Private**: All systems run locally — no external cloud dependencies

---

## 📦 Tech Stack

* **Training**: PyTorch, Vision Transformer (ViT), Kaggle Notebook
* **Model Serving**: TorchServe (`.pt` + `.mar`)
* **Experiment Tracking**: MLflow
* **Frontend**: Streamlit
* **Containerization**: Docker, Docker Compose
* **Image Handling**: Pillow, OpenCV
* **Visualization**: Matplotlib, Plotly

---

## 🛠️ Training (on Kaggle)

Training was performed on [Kaggle](https://www.kaggle.com/) using free GPU resources.

### 📍 On Kaggle:

1. Upload and run the notebook in `/training/train.py`.
2. Export the best checkpoint as `best_model.pth`.
3. Download `best_model.pth` to your local `model_store/` directory.

---

## 🔁 Convert `.pth` → `.pt` → `.mar` for TorchServe

TorchServe requires a serialized TorchScript `.pt` model and a `.mar` archive. Follow these steps:

1. **Convert `.pth` to `.pt`**

   ```bash
   python pth_to_pt.py  # outputs model_store/skin_vit.pt
   ```

2. **Create `.mar` File**

   ```bash
   torch-model-archiver \
     --model-name skin_vit \
     --version 1.0 \
     --serialized-file skin_vit.pt \
     --handler skin_lesion_handler.py \
     --extra-files "mar_config/skin_lesion_model_config.json" \
     --export-path ./ \
     --force
   ```

   This places `skin_vit.mar` and `config.properties` in the project root.

---

## 🚀 Docker & Docker Compose Deployment

We've fully containerized the system. Ensure you have **Docker Desktop** (with Docker Compose plugin) installed.

**Project Structure**

```
skin-lesion-classification/
├── config.properties        # TorchServe config
├── skin_vit.mar             # Model archive
├── docker-compose.yml
├── Dockerfile               # For Streamlit service
├── requirements.txt
├── mlflow_tracking/
│   └── mlflow_server.sh     # MLflow server launcher
├── mlruns/                  # MLflow data (local volume)
├── streamlit_app/
│   └── app.py               # Streamlit application
├── pth_to_pt.py             # Script to convert .pth to .pt
├── handlers/
│   └── skin_lesion_handler.py
├── mar_config/
│   └── skin_lesion_model_config.json
└── training/
    └── train.py
```

### ⬇️ 1. Build & Start All Containers

```bash
docker compose up --build
```

This will start three services:

* **MLflow UI** at [http://localhost:5001](http://localhost:5001)
* **TorchServe API** at [http://localhost:8080/predictions/skin\_vit](http://localhost:8080/predictions/skin_vit)
* **Streamlit App** at [http://localhost:8501](http://localhost:8501)

### ⏹️ 2. Stop & Remove Containers

```bash
docker compose down
```

### ⚙️ Docker Compose File (`docker-compose.yml`)

```yaml
version: "3.9"

services:
  mlflow:
    image: python:3.9-slim
    container_name: mlflow_server
    working_dir: /app/mlflow_tracking
    volumes:
      - ./mlflow_tracking:/app/mlflow_tracking
      - ./mlruns:/app/mlruns
    ports:
      - "5001:5001"
    command: bash mlflow_server.sh

  torchserve:
    image: pytorch/torchserve:latest
    container_name: torchserve_api
    working_dir: /app
    volumes:
      - .:/app
    ports:
      - "8080:8080"
      - "8081:8081"
    command: >
      torchserve --start \
        --model-store /app \
        --models skin_vit=skin_vit.mar \
        --ts-config /app/config.properties \
        --ncs

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: streamlit_ui
    working_dir: /app/streamlit_app
    volumes:
      - ./streamlit_app:/app/streamlit_app
    ports:
      - "8501:8501"
    depends_on:
      - mlflow
      - torchserve
    command: streamlit run app.py
```

### 🔧 Streamlit Dockerfile (`Dockerfile`)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py"]
```

---

## 🧬 Dataset

We use the [ISIC 2019 Dataset](https://challenge.isic-archive.com/data/) with:

* **25,331 training images**
* **8,232 test images**
* **9 categories**, including melanoma, nevus, and carcinoma

---

## 🔧 Custom Training

1. Place your dataset in this format:

```
data/
├── Train/
│   ├── Melanoma/
│   ├── Nevus/
│   └── ...
└── Test/
    ├── Melanoma/
    └── ...
```

2. Run training on Kaggle or locally:

```bash
cd training
python train.py
```

3. Track results at [http://localhost:5001](http://localhost:5001)
