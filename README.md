# Working with HPC
1. Login to HPC and use: `linuxsh`.
2. Go to shared folder: `cd /work3/s234843/02466-Project`
3. Activate conda with: `source /work3/s234843/bin/miniconda3/bin/activate`
4. Activate environment with: `conda activate 02466`
5. Should be all good!

# Working with Git
## dinov2-fork
In this project, we will be using a fork of dinov2 with our changes, i.e. implementation of dinov2. The fork works by acting as a "sub-repository" (aka submodule). You can download the submodule initially using:
```bash
git submodule update --init --recursive
```
(Must be ran in the root directory `02466-Project`)

After you have downloaded and other contributors have made changes, you can update the submodule by removing the `--init` argument:
```bash
git submodule update --recursive
```

# Setting up environment
DISCLAIMER: Do not do this on the HPC, it is already set up!
## dinov2-fork
The submodule already has a configured environment (`conda.yaml` or `conda-extras.yaml`), but this environment is only supported by CUDA-supported devices. Thus, we need a CUDA-device to use dinov2.

The dinov2 conda environment can be setup with:
```bash
conda env create -f dinov2-fork/conda-extras.yaml
conda activate dinov2
```


## Our Project
The normal environment can be setup with the `conda.yaml` or `cuda.yaml` located in the root directory: `02466-Project`:
```bash
conda env create -f cuda.yaml
conda activate 02466
```


# Dataset
## Setting environment variables
### macOS
1. Edit the `~/.zshrc` file
```bash
nano ~/.zshrc
```
2. Add authentication, this have been shared at some point...
```bash
export USER02466="<Username>"
export PASS02466="<Password>"
```

### Linux
1. Edit the `~/.bashrc` file
```bash
nano ~/.bashrc
```
2. Add authentication, this have been shared at some point...
```bash
export USER02466="<Username>"
export PASS02466="<Password>"
```

### Windows
```cmd
setx USER02466 "<Username>"
setx PASS02466 "<Password>"
```

## Installing dataset
1. Complete instructions above
2. Run download_dataset.py