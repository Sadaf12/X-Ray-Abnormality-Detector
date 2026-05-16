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
