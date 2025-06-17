
# 🧠 Skin Lesion Classification System



## A Complete MLOps Pipeline for Medical Image Diagnosis

This project delivers an end-to-end deep learning solution for classifying **skin lesions into 9 diagnostic categories**. It uses a fine-tuned Vision Transformer (ViT) model trained on the ISIC 2019 dataset via **Kaggle**, and integrates seamlessly with **TorchServe** for model serving, **MLflow** for experiment tracking and model versioning, and **Streamlit** for an interactive diagnostic UI.

---

## 🔍 Key Highlights

* 🧠 **ViT-based Classifier**: Trained on Kaggle using state-of-the-art techniques
* 🔁 **MLOps-Ready**: Includes model serving, tracking, and UI — all container-free and locally deployable
* 📈 **MLflow Integration**: Log, version, and monitor experiments and predictions
* 🚀 **TorchServe Deployment**: Easily serve PyTorch models as APIs
* 🌐 **Streamlit Frontend**: Drag & drop interface for quick analysis
* 🔒 **Local & Private**: All systems run locally — no cloud or external dependencies

---

## 📦 Tech Stack

* **Training**: PyTorch, Vision Transformer (ViT), Kaggle Notebook
* **Model Serving**: TorchServe
* **Experiment Tracking**: MLflow
* **Frontend**: Streamlit
* **Image Handling**: Pillow, OpenCV
* **Visualization**: Matplotlib, Plotly

---

## 🛠️ Training (on Kaggle)

Training was performed on [Kaggle](https://www.kaggle.com/) using its free GPU resources.

### To Train on Kaggle:

1. Upload your data to Kaggle or use ISIC 2019 via external URL.
2. Use the notebook in `/training/train.py` as a Kaggle Notebook script.
3. Save the best model as `best_model.pth`.
4. Download `best_model.pth` to your local `model_store/` directory for serving.

---

## 🚀 How to Run Locally

> **Note:** Make sure Python 3.8+ and Java 11+ are installed. GPU is optional but recommended.

### 🧩 1. Clone the Repo

```bash
git clone https://github.com/yourusername/skin-lesion-classification.git
cd skin-lesion-classification
```

### 📦 2. Setup Virtual Environment

```bash
python -m venv lesenv
source lesenv/bin/activate  # or lesenv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 📥 3. Place Pretrained Model

Download the trained model (`best_model.pth`) from your Kaggle training notebook and place it here:

```bash
mkdir -p model_store
# move or copy your model
cp ~/Downloads/best_model.pth model_store/
```

---

## ⚙️ Step-by-Step Execution

### ▶️ Start MLflow Tracking Server

```bash
cd mlflow_tracking
chmod +x mlflow_server.sh
./mlflow_server.sh
```

Access MLflow UI: [http://localhost:5001](http://localhost:5001)

---

### 📌 Register the Model to MLflow

```bash
cd ..
python register_model.py
```

---

### 🔥 Start TorchServe

```bash
cd torchserve
chmod +x start_torchserve.sh
./start_torchserve.sh
```

TorchServe API runs at: [http://localhost:8080/predictions/skin\_vit](http://localhost:8080/predictions/skin_vit)

---

### 🖼️ Run Streamlit Interface

```bash
cd ../streamlit_app
streamlit run app.py
```

Visit the app at: [http://localhost:8501](http://localhost:8501)

---


## 🧬 Dataset

The project uses the [ISIC 2019 Challenge Dataset](https://challenge.isic-archive.com/data/) which contains:

* **25,331 training images**
* **8,232 test images**
* **9 diagnostic labels**:

  * Melanoma
  * Melanocytic nevus
  * Basal cell carcinoma
  * Actinic keratosis
  * Benign keratosis
  * Dermatofibroma
  * Vascular lesion
  * Squamous cell carcinoma
  * None of the above

---

## 🔧 Customize / Retrain

### To Retrain Locally (Optional):

1. Place your dataset in the following format:

```
data/
├── Train/
│   ├── Melanoma/
│   ├── Nevus/
│   └── ...
└── Test/
    ├── Melanoma/
    ├── Nevus/
    └── ...
```

2. Run training:

```bash
cd training
python train.py
```

3. Monitor in MLflow at: [http://localhost:5001](http://localhost:5001)


