import os
use_gpu = True  # True=用GPU，False=用CPU
if not use_gpu:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
import sys
import re
import glob
import ast
import time
import copy
import cv2
import jax
import math
import tqdm
import shutil
import pickle
import optax
import jaxopt
import socket
from pathlib import Path
from jax import random, jit, vmap, value_and_grad, jvp
import jax.numpy as jnp
from jax import lax
from jax import vmap
from jax.tree_util import tree_map
import jax.nn
import scipy.stats
import numpy as np
from PIL import Image
from functools import partial
from math import factorial
from typing import List, Tuple, Dict, NamedTuple
from sklearn.cluster import KMeans
from types import SimpleNamespace
from scipy.ndimage import label
import scipy.io as sio
from scipy.io import loadmat
from scipy.io import savemat
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.collections as mcoll
import IPython.display
from tensorboardX import SummaryWriter

def seed_everything(seed_value: int):
    """
    Fix random seeds for NumPy and JAX.
    Returns
    -------
    key : jax.random.PRNGKey
        Base JAX random key (should be split explicitly later).
    """
    # ---------- NumPy ----------
    np.random.seed(seed_value)
    # ---------- JAX ----------
    key = jax.random.PRNGKey(seed_value)
    return key

seed = 42
seed_everything(seed)
jax.clear_caches()
