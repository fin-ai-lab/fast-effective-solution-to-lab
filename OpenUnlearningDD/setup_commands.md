wget https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Linux-x86_64.sh
sh Anaconda3-2025.06-1-Linux-x86_64.sh
exit

conda create -n unlearning python=3.11
conda activate unlearning
pip install .[lm_eval]
pip install --no-build-isolation flash-attn==2.6.3
huggingface-cli login