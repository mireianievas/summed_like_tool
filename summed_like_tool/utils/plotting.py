import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

def new_viridis():
    viridis = mpl.cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[:25, :]*=np.repeat(np.linspace(0,1,25),4).reshape(-1,4)
    newcolors[:25, 3] = 1
    new_viridis = mpl.colors.ListedColormap(newcolors)
    return(new_viridis)