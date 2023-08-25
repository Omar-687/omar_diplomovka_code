import torch
from torch import nn
import torch.optim as optim
import numpy as np
import json
import matplotlib.pyplot as plt
from util import *
from controller2 import *

dim_in = 1
dim_hid = 1
dim_out = 1
min_timestamp = 1
model1 = PPCController(dim_in, dim_hid, dim_out,  3*10**-4,min_timestamp)



L = 1
T = 1
N = 2
sts = np.array([[[1, 1]]])
ej = np.array([1, 1])

# Calculate MPE using original_mpe function
mpe = model1.mpe(sts, ej, L, T, N)




# Check that MPE is within bounds
# assert mpe >= 0 and mpe <= 1, f"MPE value of {mpe} is outside expected bounds"
print(f'First MPE test: {mpe}')


L = 1
T = 2
N = 2
sts = np.array([[[1, 1],[0, 0]]])
ej = np.array([1, 1])

# Calculate MPE using original_mpe function
mpe = model1.mpe(sts, ej, L, T, N)
print(f'Second MPE test: {mpe}')



L = 1
T = 2
N = 2
sts = np.array([[[0.5, 0.75],[0.5, 0.25]]])
ej = np.array([1, 1])

# Calculate MPE using original_mpe function
mpe = model1.mpe(sts, ej, L, T, N)
print(f'Third MPE test: {mpe}')



# least_laxity_scheduling(t_arr)

# TRAINING

model = PPCController(dim_in=dim_in, dim_hid=dim_hid, dim_out=dim_out, alpha=3*10**-4, min_timestamp=min_timestamp);
# model.my_train(input1,r_j)
model.training_experimental()