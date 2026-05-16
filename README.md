# X-Ray Abnormality Detector

Streamlit application and MLOps pipeline for classifying chest X-ray images as normal or abnormal.

## Deployed Application

The deployed web application is available here:

https://xray-ui-402244708304.europe-west6.run.app/

The Cloud Run UI service is limited to a maximum of 2 instances to reduce cloud costs, so it may be slow or temporarily unavailable under higher traffic.

## Environment Setup

### Requirements

- Python 3.11
- Git
- DVC with GCS support, installed through `requirements.txt`
- Google Cloud credentials if you want to pull data, upload metrics, or read model metadata from GCS

### Create virtual environment

Windows (PowerShell):

```bash
py -3.11 -m venv .venv_xray
.\.venv_xray\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv .venv_xray
source .venv_xray/bin/activate
```

### Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The default requirements install CUDA-enabled PyTorch wheels for NVIDIA GPUs. CPU-only or non-NVIDIA environments may need a different PyTorch installation command from the official PyTorch selector.

### Verify installation

```bash
python --version
pip show torch
```

## Data Setup

The DVC remote is configured in `.dvc/config`. To restore the tracked data locally:

```bash
dvc pull
```

This restores the data artifacts, including:

```text
data/raw/nih_chest_xray
data/processed/binary_labels.csv
```

## Local Application

Start the API in one terminal:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Start the Streamlit UI in another terminal:

```bash
streamlit run ui/streamlit_app.py
```

The UI opens at:

```text
http://localhost:8501
```

For local GCS access, set Google Application Default Credentials or point to a service account key:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
$env:DATA_BUCKET="nih-xray-data"
```

Do not commit service account JSON keys to Git.

## Training and Evaluation

The main training entry point is:

```bash
python src/training/train.py
```

Supported modes:

```text
RUN_MODE=train      trains the model, evaluates the held-out test set, and uploads model/evaluation artifacts
RUN_MODE=evaluate   skips training, evaluates an existing checkpoint, and uploads evaluation artifacts
RUN_MODE=hparam     runs Optuna hyperparameter search
```

Example: evaluate an existing local checkpoint without retraining:

```powershell
$env:RUN_MODE="evaluate"
$env:DATA_BUCKET="nih-xray-data"
$env:MODEL_LOCAL_PATH="models/best_model.pt"
python src/training/train.py
```

Evaluation artifacts are uploaded to:

```text
gs://<DATA_BUCKET>/models/evaluation/test_metrics.json
gs://<DATA_BUCKET>/models/evaluation/roc_curve.json
```

The Streamlit UI reads these files and displays the current model performance in the right-side panel.

## Cloud Build / Deployment

Training image build:

```bash
gcloud builds submit --config cloudbuild.yaml .
```

Launch Vertex AI training job:

```bash
python scripts/launch_vertex_job.py
```

Deploy API:

```bash
gcloud builds submit --config cloudbuild-api.yaml .
```

Deploy UI:

```bash
gcloud builds submit --config cloudbuild-ui.yaml .
```
