# Biomedical image analysis: NK-cell segmentation

> **TL;DR:** Deep-learning segmentation of NK cells in microscopy images, comparing an OpenCV baseline with ImageNet- and DINO-pretrained U-Net models. I contributed training and cross-validation code, image/mask augmentation, evaluation and statistical comparisons.

## My contribution - Alfred Aqraou

- Developed and refined cross-validation and hyperparameter-search workflows, including nested cross-validation, multiprocessing and experiment logging.
- Implemented and corrected joint image/mask transformations and per-sample seeding so geometric augmentations remain aligned and reproducible.
- Added DINO-initialized U-Net support, IoU/Jaccard-loss work and evaluation outputs for comparing model configurations.
- Built per-image IoU analysis, confidence-interval and boxplot comparisons, Friedman tests and predicted-versus-ground-truth visualizations.

This was a collaborative DTU project. The original project documentation below reports the team's results; my individual work is summarized above.

**Original team repository:** [MathiasKES/02466-Project](https://github.com/MathiasKES/02466-Project). This portfolio fork preserves the original history and attribution.

**Selected contributions:** [Nested cross-validation and reproducibility](https://github.com/MathiasKES/02466-Project/commit/21ca34f33618b7e29aac6103714b0f550e6e79f3) · [DINO-initialized U-Net](https://github.com/MathiasKES/02466-Project/commit/93a637e17729d5c6915f784765122f57d7bac884) · [Statistical model comparisons](https://github.com/MathiasKES/02466-Project/commit/98c926566a3fe9c8b1fc4df1a2fe967a0107af33).

---

## Project documentation

<img src="illustrations/gifs/transformed_fluorescent_grid_612x445.gif" alt="Fluorescent NK Cell Animation" />

This repository contains the implementation of a deep learning project for segmenting Natural Killer (NK) cells in fluorescence microscopy images, developed as part of the *02466: Project Work - Bachelor of Artificial Intelligence and Data* course at the Technical University of Denmark (DTU) in Spring 2025. The project compares three segmentation approaches: a traditional OpenCV-based baseline, a supervised U-Net model pretrained on ImageNet, and a self-supervised U-Net model pretrained using the DINO framework. The codebase leverages PyTorch, solo-learn, and DTU’s High Performance Computing (HPC) cluster for reproducible experiments, hyperparameter tuning, and performance evaluation.

## Project Overview

The project addresses automated segmentation of NK cells in a dataset of 16,038 microscopy images (8,019 broadband, 8,019 fluorescent) of human prostate cancer cells (DU145) exposed to NK cells and antibody treatments to study Antibody-Dependent Cellular Cytotoxicity (ADCC). With only 41 labeled fluorescent images, transfer learning and data augmentation mitigate data scarcity. The supervised U-Net (ResNet50 encoder, ImageNet pretrained) achieved a mean Intersection over Union (IoU) of 80.35%, the self-supervised U-Net (ResNet18 encoder, DINO pretrained) achieved 78.85%, and the OpenCV baseline achieved 62.49%. The codebase supports nested cross-validation, visualization, and statistical analysis, contributing to SDG 8 (Decent Work and Economic Growth) by automating biomedical image analysis.

### Code Summary

The main script (`unet/nested_CV.py`) implements a nested cross-validation pipeline for training and evaluating U-Net models. Key features include:

- **Dataset Handling**: `NKCellDataset` class loads images and masks, with joint transformations (resizing, flipping, rotation) for augmentation.
- **Model Architecture**: U-Net models with ResNet18/34/50 encoders (ImageNet pretrained) or ResNet18 (DINO pretrained via solo-learn), configurable via command-line arguments.
- **Hyperparameter Tuning**: Nested cross-validation optimizes batch size, loss function (Dice, IoU, Combined), optimizer (Adam, SGD), learning rate, encoder type, encoder freezing, and resolution.
- **Training and Evaluation**: Training with early stopping, per-image IoU computation, and visualization of predicted vs. ground truth masks.
- **Parallel Processing**: Multiprocessing for hyperparameter searches, with `tqdm` for progress tracking.
- **Output**: Results saved as CSV files, per-image IoU scores as NumPy arrays, and visualizations of top configurations.

The code ensures reproducibility with fixed seeds (`SEED=42`), detailed logging, and timestamped exports.

## Usage

Example run of the training script with configurable hyperparameters:
```bash
python unet/nested_CV.py \
  --epochs 100 \
  --encoders resnet18 resnet50 dino/path/to/ckpt.ckpt \
  --loss dice iou combined \
  --batchsizes 4 8 \
  --freeze True False \
  --optimizers adam \
  --lr 0.0001 0.0005 \
  --resize 256 512 \
  --outer 3 \
  --inner 3 \
  --workers 8
```

### Arguments
- `--workers`: CPU workers for parallel processing (default: 8).
- `--epochs`: Training epochs.
- `--encoders`: Encoders (e.g., `resnet18`, DINO checkpoint path).
- `--loss`: Loss functions (`dice`, `iou`, `combined`, `bce`).
- `--batchsizes`: Batch sizes (e.g., `4 8`).
- `--freeze`: Freeze encoder (`True`/`False`).
- `--optimizers`: Optimizers (`adam`, `sgd`).
- `--lr`: Learning rates (e.g., `0.0001 0.0005`).
- `--resize`: Resolutions (e.g., `256 512`).
- `--outer`: Outer cross-validation folds.
- `--inner`: Inner cross-validation folds.

### Outputs
Results are saved to `exports/unet/mm-dd/HH-MM-SS`:
- `results.csv`: Mean/std of test loss and IoU per configuration.
- `*.npy`: Per-image IoU scores.
- `*.png`: Predicted vs. ground truth masks, top configurations.
- `runtime.log`: Fold indices and runtime logs.

## HPC
## Accessing the HPC
The first time accessing the HPC cluster can be tricky for inexperienced users, this guide will aim to simplify this.

Initially, the connection to HPC must be done on either a VPN to DTU, or directly on DTU's network. Accessing the HPC can be done with either a GUI (such as ThinLinc) or SSH (recommended), which is explained in this guide. After the initial connection, a "ssh-key" can be set up to allow access from remote places without the use of a VPN.

### 1) Creating SSH-key for authentication
To access the HPC outside of DTU without the use of a VPN, you have to create a so-called 'SSH-key', which you (preferably) should generate uniquely on each of the devices you plan to use HPC from. Start by opening a terminal in your user profile, then create a `.ssh` directory and move into it, and lastly generate the key:
```bash
cd ~/
mkdir -p .ssh
cd .ssh
ssh-keygen -t ed25519 -f dtuhpc
```
When prompted for a password you can simply leave this empty by pressing enter twice. This generates a private and a public (`.pub`) key. You should never share your private key - only the public since this key functions as a "handshake" between your device and the HPC. The public key has now been created as `~/.ssh/dtuhpc.pub`. Print the contents of this file and copy it manually:
```bash
cat ~/.ssh/dtuhpc.pub
```
This one line is your public key, copy it for later.

### 1.5) Connecting with VPN
Connecting from outside DTU's network is almost like connecting from inside, you just need an additional step. Firstly, on a personal computer, install a VPN client such as [AnyConnect](https://net.ait.dtu.dk/vpn/), open it. Type in `vpn.dtu.dk`, click `Connect`, enter your DTU credentials and lastly your DTU 2-Factor Authentication. Now you are ready to access the HPC.

### 2) Accessing from DTU network
The LSF 10 cluster can be accessed through 4 different login-nodes:  
`<userid>@login1.gbar.dtu.dk`  
`<userid>@login1.hpc.dtu.dk`  
`<userid>@login2.gbar.dtu.dk`  
`<userid>@login2.hpc.dtu.dk`  

The `<userid>` must be replaced by your DTU student id, e.g. `s234843@login1.hpc.dtu.dk`. I usually just use this login node. Use ssh to login:
```bash
ssh -i ~/.ssh/dtuhpc <userid>@login1.hpc.dtu.dk
```
The `-i <private key>` argument specifies the location of your newly generated private key. If prompted for password, use your DTU login. If you created a password for the ssh-key, this usually comes first.

### 3) Switching node
You should now be signed into the HPC on their login node, switch to a "work" node by executing:
```bash
linuxsh
```

### 4) Setting up SSH-key
Once you're in, you can go to your user profile on the HPC (this is not the same user profile as from step 1).
```bash
cd ~/.ssh
```
You must now add your public key to the `authorized_keys` file on the HPC, you can enter a text editor using:
```bash
nano authorized_keys
```
Paste your public key from step 1) by pressing: `Ctrl+Shift+V`. Now save and exit by pressing `Ctrl+X`, followed by `Y` and `Enter`. Now you should be able to connect to the HPC without VPN. If you plan on adding more devices to connect from, simply add a new line and paste that public key, then save and exit.

Continue to next section [Working with HPC](#working-with-hpc).

## Working with HPC
1. Login to HPC and use: `linuxsh`.
2. Go to shared folder: `cd /work3/s234843/02466-Project`

(If you cannot access this directory, contact Mathias - dataset is only available group members)

3. Activate environment with: 
```bash
source /work3/s234843/init_project.sh
```
For your convenience copy this file to your user profile so you can run it directly when logging in on the HPC:
```bash
cp /work3/s234843/init_project.sh ~/init_project.sh
```
From now on you can run with:
```bash
source ~/init_project.sh
```

### ⚠️ **IMPORTANT** ⚠️
Apparently our PyTorch version only supports `GPU compute capability >= 7.5`, but GPU's such as the Tesla V100 has version 7.0, which means that the only available GPU's for our project are:
```
gpua100
gpua10
gpua40
```

### Creating batch jobs for GPU usage
Check out some of the `.sh` files for working examples.
#### BSUB options:
[For reference](https://www.hpc.dtu.dk/?page_id=1416)

These can also be run as an argument when submitting a job, ie. `bsub -J <job_name> < <file>`
- Specify queue: `#BSUB -q <gpu>` (see list of available gpu's [here](https://www.hpc.dtu.dk/?page_id=2759))
- Set job name: `#BSUB -J <job_name>`
- Set output log: `#BSUB -o <file>` (add extra o for overwriting previous file contents)
- Set error log: `#BSUB -e <file>`
- Request memory: `#BSUB -R "rusage[mem=4096]"`
- Specify walltime: `#BSUB -W <hh:mm>`
- Number of cores: `#BSUB -n <num_cores>`
- Set job dependency: `#BSUB -w "done(<job_id>)"` (run after another job finishes)
- Send email on job events: `#BSUB -u <email>` and `#BSUB -B -N` (`-B` email at start, `-N` email at end)
- Set project/account: `#BSUB -P <project_name>`
- Set job to restartable: `#BSUB -r`
- Request GPU resources: `#BSUB -gpu "num=1"` (or other appropriate GPU spec)

### bsub commands
- Submit a job: `bsub < <file>`
- Submit inline command: `bsub <command>` 
- Track job: `bstat`
- Killing job: `bkill <JOBID>`
- Show job output: `bpeek <JOBID>`
- View job history: `bhist -u <username>`
- Viewing job queue: `bjobs`
- Viewing detailed job info: `bjobs -l <JOBID>`
- Viewing HPC gpu status: `bqueues`




### Transfering large files to HPC
It is instructed to use their transfer nodes instead of login nodes for faster and perhaps more stable transferral.
```bash
scp -i ~/.ssh/dtuhpc <local file> <userid>@transfer.gbar.dtu.dk:<hpc location>
```

## Working with Git
### Setting git user config
Initially, you must give git some information about yourself:
```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```
Continue to [Setting up ssh](#setting-up-ssh).

### Setting up ssh
Using git for the first time on hpc requires that you set up an ssh key that is recognized by your personal GitHub profile. Here is a step by step guide for doing this:
1. Create a github key on your HPC user and copy the output from `cat github.pub` command:
```bash
cd ~/.ssh
ssh-keygen -t ed25519 -C "your_email@example.com" -f github
cat github.pub
```
2. [Add the key to GitHub](https://github.com/settings/ssh/new). The title you can just call `DTU HPC` and in the key input field you must insert the output copied from the `cat github.pub` command. Click `Add SSH key`.
3. Back in the HPC terminal, run:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github
```
And for extra measures do the following:
```bash
nano ~/.ssh/config
```
Write in:
```bash
Host github.com
  User git
  IdentityFile ~/.ssh/github
```
Close and save with `CTRL + X`, then `y`, then `Enter`.

You should be done now and can begin working with git!

### solo-learn
In this project, we will be using a fork of solo-learn with our configuration changes. The fork works by acting as a "sub-repository" (aka submodule). You can download the submodule initially using:
```bash
git submodule update --init --recursive
```
(Must be ran in the root directory `02466-Project`)

After you have downloaded and other contributors have made changes, you can update the submodule by removing the `--init` argument:
```bash
git submodule update --recursive
```

## Setting up environment
DISCLAIMER FOR MEMBERS OF THE GROUP: Do not do this on the HPC, it is already set up!
### Our Project
The normal environment can be setup with the `cpu.yaml` or `cuda.yaml` located in the root directory: `02466-Project`:
```bash
conda env create -f cuda.yaml
conda activate 02466_cuda
```
or for cpu:
```bash
conda env create -f cpu.yaml
conda activate 02466_cpu
```

If the yaml file was updated since you installed it, you can update it using:
```bash
conda env update --file <file>.yaml --prune
```

### solo-learn
The submodule works with our conda environments described in [Our Project](#our-project).

## Dataset
### Setting environment variables
#### macOS
1. Edit the `~/.zshrc` file
```bash
nano ~/.zshrc
```
2. Add authentication, this have been shared at some point...
```bash
export USER02466="<Username>"
export PASS02466="<Password>"
```

#### Linux
1. Edit the `~/.bashrc` file
```bash
nano ~/.bashrc
```
2. Add authentication, this have been shared at some point...
```bash
export USER02466="<Username>"
export PASS02466="<Password>"
```

NOTE: If you are planning to use the HPC, you should also do this for yourself with the same `~/.bashrc` file as above. This file is personal for your own user on the HPC cluster.

#### Windows
```cmd
setx USER02466 "<Username>"
setx PASS02466 "<Password>"
```

### Installing dataset
1. Complete instructions above
2. Run download_dataset.py

## Useful commands
### Giving access to directory (for Mathias):
```bash
chgrp -R 10196 /work3/s234843/02466-Project &
chgrp -R 10196 /work3/s234843/bin &
chmod -R g+rwxs /work3/s234843/02466-Project &
chmod -R g+rwxs /work3/s234843/bin &
```

## Reproducibility
- Use `SEED=42` for consistent results.
- Ensure dataset access at `/work3/s234843/02466-Project/dataset`.
- Replicate the environment with `requirements.txt` or `cuda.yaml`.
- Run on DTU HPC for large hyperparameter searches.

## Limitations
- Optimized for DTU’s dataset and HPC, requiring adaptation for other datasets.
- Memory constraints limit large batch sizes/high resolutions.
- DINO pretraining requires checkpoint files or solo-learn re-implementation.

## Future Work
- Support additional datasets.
- Explore Vision Transformers.
- Implement Bayesian hyperparameter optimization.
- Optimize for non-HPC environments.

## Acknowledgments
We express our deepest gratitude to our supervisors for their invaluable guidance and support throughout this project. Their expertise in machine learning and biomedical imaging was instrumental in shaping our approach and achieving our objectives. We also thank DTU’s Department of Biotechnology and Biomedicine for providing the NK cell dataset, the HPC cluster team for computational resources, and the DigitSTEM project for baseline insights. Finally, we acknowledge the open-source community for tools like PyTorch and solo-learn, which were critical to our success.