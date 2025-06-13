import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'unet')))
from multiprocessing import Pool, set_start_method
import csv
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from itertools import product
from sklearn.model_selection import KFold
from collections import defaultdict
from datetime import datetime
from functools import reduce
import operator
import re
from typing import List
import pandas as pd
import wandb
from dataset import NKCellDataset, JointTransform
from models.unet import create_unet
from models.unet import create_unet_with_dino_resnet18
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss
import argparse
import random