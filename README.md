# Quant ML Modeling Framework
An end-to-end quantitative machine learning framework designed for financial time-series prediction. This repository specializes in identifying and forecasting market structural pivot points—specifically **peaks** and **troughs**—using a dual-engine architecture powered by both classical machine learning (scikit-learn) and deep learning (PyTorch & PyTorch Lightning).
Additionally, the framework features probability calibration mechanisms and native **ONNX pipeline migration** utilities for low-latency, production-ready model deployment.
## 🚀 Key Features
 * **Dual-Engine Pipelines:** Complete separation of concerns between scikit-learn and PyTorch execution pipelines, allowing for seamless benchmarking.
 * **Peak & Trough Structural Labeling:** Dedicated data engineering components engineered specifically to ingest, label, and process financial indicators into distinct target states.
 * **Probability Calibration:** Custom neural network architectures implementing an integrated Isotonic Layer for output calibration, ensuring predicted model confidences align with true empirical probabilities.
 * **Production Deployment via ONNX:** Compilation scripts built-in to seamlessly export and test trained PyTorch and Scikit-learn models into cross-platform ONNX formats.
 * **Modern Tooling & Linting:** Strictly maintained codebase utilizing modern, ultra-fast Python linting configurations via ruff.
## 📂 Repository Structure
The architecture is divided into localized pipeline utilities (src/), direct modeling entry points (run/), and targeted framework sub-directories (scikit_models/ & torch_models/):
```text
quant-ml-modeling/
├── run/                     # Top-level script runners
│   ├── plot/                # Model output visualization scripts
│   ├── sklearn/             # Scikit-learn training scripts (Peak & Trough MLPs)
│   └── torch/               # PyTorch training, loading, and evaluation workflows
├── scikit_models/           # Scikit-learn specific sub-framework
│   └── src/                 
│       ├── model/           # Multi-Layer Perceptron architecture definitions
│       ├── inference/       # Prediction and model execution APIs
│       └── run/             # Scikit-learn localized execution modules
├── torch_models/            # PyTorch / Lightning specific sub-framework
│   └── src/                 
│       ├── data/            # PyTorch Dataset wrappers and data loaders
│       ├── model/           # Deep MLP, Calibrated MLP, and Isotonic Layers
│       ├── inference/       # PyTorch model scoring pipelines
│       └── run/             # PyTorch execution loops
├── src/                     # Shared core utilities 
│   ├── data/                # Loaders, engineering pipelines, and labeling modules
│   ├── scalers/             # Custom MinMax, Standard, and Base Scaling classes
│   └── visualization/       # Plotting engines for data analysis
├── utility/                 # ONNX translation and cross-model evaluation utilities
├── pyproject.toml           # Global project build and metadata definitions
├── ruff.toml                # Linting and style enforcement settings
├── requirements_sklearn.txt # Classical ML dependencies
└── requirements_torch.txt   # Deep Learning and hardware-acceleration dependencies

```
## 🛠️ Installation & Setup
Because this project accommodates both traditional machine learning models and deep neural networks, dependencies are split into specialized environment configurations.
### 1. Clone the Repository
```bash
git clone https://github.com/ShaahinToloo/quant-ml-modeling.git
cd quant-ml-modeling

```
### 2. Configure Your Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```
### 3. Install Dependencies
Choose the configuration file that matches your experimentation goals, or install both for full system functionality:
 * **For Scikit-learn Workflows:**
   ```bash
   pip install -r requirements_sklearn.txt
   ```
* **For PyTorch / Deep Learning Workflows:**
  ```bash
  pip install -r requirements_torch.txt

  ```
## 📊 Workflow & Usage
The repository uses highly modular execution scripts. You can run training or inference pipelines directly from your terminal.
### Classical ML Modeling (Scikit-Learn)
To train the scikit-learn based Multi-Layer Perceptrons for market extrema tracking:
```bash
# To train peak predictive modeling
python run/sklearn/mlp_peak.py

# To train trough predictive modeling
python run/sklearn/mlp_trough.py

```
### Deep Learning Modeling (PyTorch & Lightning)
To trigger PyTorch Lightning-backed neural network training sequences:
```bash
# Train deep learning engines from scratch
python run/torch/torch_mlp_peak.py
python run/torch/torch_mlp_trough.py

# Load pre-trained neural configurations for evaluation
python run/torch/torch_mlp_peak_load.py

```
### Visualization and Output Mapping
To analyze performance boundaries, map outputs, and view localized peak/trough predictions:
```bash
python run/plot/plot_sklearn_models_output.py
python run/plot/plot_torch_models_output.py

```
## 🚀 Production Deployment & ONNX Compilation
To transition model weights from research to live production execution feeds without incurring Python runtime latency penalties, compile your tracking architectures directly to an ONNX runtime environment:
```bash
# Export active model architectures to static ONNX binaries
python utility/export_onnx.py

# Run verification and latency validation checks on exported ONNX objects
python utility/test_onnx_models.py
```
> [!CAUTION]
> When you need to convert a model to onnx, MAKE SURE your DO NOT add conditions in 'def forward' of the model.
> Why? Because ONNX runs the model and traces it's paths of calculation.
> So if you add conditions like '''If X[0] > 16: ...''' in there, then 💀.

## 📝 License
This project is open-source software licensed under the **MIT License**. See the LICENSE file for more details.
