Adaptive-SGD

🧬 MultiModal Nonconvex Optimizer (RNA–ATAC–CITE)

Digital Twin for Cellular Reprogramming
This project implements a multi-modal optimization framework that integrates RNA, ATAC, and CITE-seq modalities using a nonconvex Adaptive Stochastic Gradient Descent (ASGD) optimizer.
It supports single-cell perturbation, cell state reprogramming, and the creation of a digital twin of wet-lab differentiation experiments.

🚀 Mission
Our mission is to accelerate recovery from cancer and combat autoimmune diseases through advanced cellular reprogramming and the development of a digital twin of stem cell differentiation.
This framework allows bioinformaticians to simulate, optimize, and refine cellular differentiation processes in silico, enabling faster discoveries and more personalized therapeutic strategies.

⚙️ Environment Setup

1️⃣ Create and activate environment

conda create -n multimodal python=3.10 -y

conda activate multimodal

2️⃣ Install dependencies

pip install torch torchvision torchaudio

pip install numpy pandas scikit-learn scipy matplotlib seaborn

pip install scanpy anndata pytorch-lightning tqdm jupyter

If using GPU acceleration:

pip install torch --index-url https://download.pytorch.org/whl/cu121



🧠 Reproduction Steps

Launch Jupyter

jupyter notebook

Open Notebook

MultiModal_Nonconvex_Optimizer(RNA-ATAC-CITE modalities)-updated.ipynb

Run All Cells

Generates embeddings for each modality

Applies adaptive nonconvex optimization

Produces evaluation metrics and visualizations

Benchmark Optimizers

Automatically compares:

Adam

AMSGrad

ASGDAdam (proposed)

ASGDAMSgrad(proposed)

Padam

SGD with momentum

📊 Outputs

File	Description

results/metrics.csv	Summary of accuracy, precision, F1, R², rmse, mse, mae



🧪 Key Features

✅ Multi-modal RNA–ATAC–CITE integration

✅ Nonconvex curvature-aware optimizer

✅ Dynamic learning rate switching

✅ Single-cell perturbation simulation

✅ Digital twin for cell differentiation modeling
