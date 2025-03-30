# HPC
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

## Working with HPC
1. Login to HPC and use: `linuxsh`.
2. Go to shared folder: `cd /work3/s234843/02466-Project`
3. Activate conda with: `source /work3/s234843/bin/miniconda3/bin/activate`
4. Activate environment with: `conda activate 02466`
5. Should be all good! Now you can run scripts manually.

### Creating batch jobs for GPU usage
Check out some of the `.sh` files for working examples.
#### BSUB options:
[For reference](https://www.hpc.dtu.dk/?page_id=1416)
- Specify queue: `#BSUB -q <gpu>` (see list of available gpu's [here](https://www.hpc.dtu.dk/?page_id=2759))
- ...

### bstat commands
- Viewing job queue: `bstat`
- Killing job: `bkill <JOBID>`


### Transfering large files to HPC
It is instructed to use their transfer nodes instead of login nodes for faster and perhaps more stable transferral.
```bash
scp -i ~/.ssh/dtuhpc <local file> <userid>@transfer.gbar.dtu.dk:<hpc location>
```

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

NOTE: If you are planning to use the HPC, you should also do this for yourself with the same `~/.bashrc` file as above. This file is personal for your own user on the HPC cluster.

### Windows
```cmd
setx USER02466 "<Username>"
setx PASS02466 "<Password>"
```

## Installing dataset
1. Complete instructions above
2. Run download_dataset.py