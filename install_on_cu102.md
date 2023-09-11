conda create -n jamil python=3.10 ipykernel
conda activate jamil

# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

pip install torchtext -i https://download.pytorch.org/whl/cu102/

conda install --file requirements.txt -c huggingface