## Installation
Modified from [CenterPoint](https://github.com/tianweiy/CenterPoint/tree/5b0e574a4478086ee9686702456aaca4f4115caa)'s original document.

### Requirements

- Linux
- Python 3.6+
- PyTorch 1.8.1 (We only test on 1.8.1)
- CUDA 11.1 or higher
- CMake 3.13.2 or higher
- [APEX](https://github.com/nvidia/apex)
- [spconv](https://github.com/traveller59/spconv/commit/73427720a539caf9a44ec58abe3af7aa9ddb8e39) 

#### Notes
- spconv should be the specific version from links above
- The spconv version after this commit will consume much more memory. 
- A rule of thumb is that your pytorch cuda version must match the cuda version of your systsem for other cuda extensions to work properly. 

we have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04/18.04
- Python: 3.6.5
- PyTorch: 1.8.1
- CUDA: 11.1
- CUDNN: 8.2.0

### Basic Installation 

```bash
# basic python libraries
conda create --name centerpoint python=3.8
conda activate sparse2dense
conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=11.1 -c pytorch
git clone https://github.com/stevewongv/sparse2dense.git
cd Sparse2Dense
pip install -r requirements.txt

# add CenterPoint to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_CENTERPOINT"
```

### Advanced Installation 


#### Cuda Extensions

```bash
# set the cuda path(change the path to your own cuda location) 
export PATH=/usr/local/cuda-11.1/bin:$PATH
export CUDA_PATH=/usr/local/cuda-11.1
export CUDA_HOME=/usr/local/cuda-11.1
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
bash setup.sh 
```

#### APEX

```bash
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 5633f6  # recent commit doesn't build in our system 
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

#### spconv
```bash
sudo apt-get install libboost-all-dev
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *
```

#### Check out [GETTING_START](GETTING_START.md) to prepare the data and play with all those pretrained models. 