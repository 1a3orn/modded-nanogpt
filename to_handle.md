pip install numpy tqdm torch huggingface-hub

pip install --pre torch==2.7.0.dev20250110+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade

apt-get update && apt-get install -y build-essential

SCALE_FACTOR=32 ./run.sh


# To Do on the GPU
- layer_definitions = ["mlp"] * 4 + ["mlp_moe"] + ["mlp"] * 2 + ["mlp_moe"] + ["mlp"] * 4