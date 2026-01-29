import os
from utils_functions import global_vars

os.system('conda env_emi_burden create -f env.yml')
os.system('conda activate env_emi_burden')


try:
    os.makedirs(global_vars.plot_dir)
except OSError:
    pass

try:
    os.makedirs('outputs')
except OSError:
    pass

