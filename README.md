

# 🧠 Skin Lesion Classification System


## A Complete MLOps Pipeline for Medical Image Diagnosis

This project delivers an end-to-end deep learning solution for classifying **skin lesions into 9 diagnostic categories**. It uses a fine-tuned Vision Transformer (ViT) model trained on the ISIC 2019 dataset via **Kaggle**, and integrates seamlessly with **TorchServe** for model serving, **MLflow** for experiment tracking and model versioning, and **Streamlit** for an interactive diagnostic UI.

---

## 🔍 Key Highlights

* 🧠 **ViT-based Classifier**: Trained on Kaggle using state-of-the-art techniques
* 🔁 **MLOps-Ready**: Includes model serving, tracking, and UI — all container-free and locally deployable
* 📈 **MLflow Integration**: Log, version, and monitor experiments and predictions
* 🔥 **TorchServe API**: Serve the model via REST API using `.mar` file
* 🌐 **Streamlit Frontend**: Drag & drop interface for quick analysis
* 🔒 **Local & Private**: All systems run locally — no cloud or external dependencies

---

## 📦 Tech Stack

* **Training**: PyTorch, Vision Transformer (ViT), Kaggle Notebook
* **Model Serving**: TorchServe (`.pt` + `.mar`)
* **Experiment Tracking**: MLflow
* **Frontend**: Streamlit
* **Image Handling**: Pillow, OpenCV
* **Visualization**: Matplotlib, Plotly

---

## 🛠️ Training (on Kaggle)

Training was performed on [Kaggle](https://www.kaggle.com/) using its free GPU resources.

### 📍 On Kaggle:

1. Upload and run the notebook in `/training/train.py`.
2. Export the best checkpoint as `best_model.pth`.
3. Download `best_model.pth` to your local `model_store/` directory.

---

## 🔁 Convert `.pth` → `.pt` → `.mar` for TorchServe

TorchServe requires a serialized `.pt` TorchScript model and a `.mar` archive. Here's how to do it:

### ✅ 1. Convert `.pth` to `.pt`

Use the script `pth_to_pt.py` (you must create it or add to your pipeline):



Run it:

```bash
python pth_to_pt.py
```

---

### ✅ 2. Create `.mar` File

TorchServe needs the following:

* `skin_vit.pt` – TorchScript model
* `skin_lesion_handler.py` – custom handler
* `mar_config/skin_lesion_model_config.json` – metadata config

Then run:

```bash
torch-model-archiver \
  --model-name skin_vit \
  --version 1.0 \
  --serialized-file model_store/skin_vit.pt \
  --handler handlers/skin_lesion_handler.py \
  --extra-files "mar_config/skin_lesion_model_config.json" \
  --export-path model_store/ \
  --force
```

This will generate:

```
model_store/
└── skin_vit.mar
```

Now you're ready to serve it!

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

Download the trained `.pth` from Kaggle and place it in:

```bash
model_store/best_model.pth
```

Then convert to `.pt` and `.mar` using steps above.

---

## ⚙️ Step-by-Step Execution

### ▶️ Start MLflow Tracking Server

```bash
cd mlflow_tracking
chmod +x mlflow_server.sh
./mlflow_server.sh
```

Access: [http://localhost:5001](http://localhost:5001)

---

### 📌 Register Model to MLflow

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

TorchServe API: [http://localhost:8080/predictions/skin\_vit](http://localhost:8080/predictions/skin_vit)

---

### 🖼️ Launch Streamlit Interface

```bash
cd ../streamlit_app
streamlit run app.py
```

Streamlit App: [http://localhost:8501](http://localhost:8501)

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

