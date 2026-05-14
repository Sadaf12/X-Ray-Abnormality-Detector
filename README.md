## Environment Setup

### Requirements

- Python 3.11
- Git

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

### Verify installation

```bash
python --version
pip show torch
```