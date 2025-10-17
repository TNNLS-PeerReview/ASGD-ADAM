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

🧩 Parameter Settings

Parameter	Description	Default

lr	Base learning rate	1e-3

beta1, beta2	Momentum coefficients	0.9, 0.999

eps	Stability constant	1e-8

switch_interval	Learning rate switching frequency (ASGD)	10

alpha_min, alpha_max	Min/max adaptive rates	1e-4, 1e-2

batch_size	Mini-batch size	128

epochs	Number of training epochs	50

optimizer	Optimizer choice (Adam, AMSGrad, ASGDAdam)	ASGDAdam

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

📊 Outputs

File	Description

results/metrics.csv	Summary of accuracy, precision, F1, R²

results/loss_curves.png	Training/validation loss curves

results/landscape.png	Nonconvex loss landscape visualization

results/final_model.pt	Trained model weights

🧪 Key Features

✅ Multi-modal RNA–ATAC–CITE integration

✅ Nonconvex curvature-aware optimizer

✅ Dynamic learning rate switching

✅ Single-cell perturbation simulation

✅ Digital twin for cell differentiation modeling
