import pandas as pd

from utils import *
from mlpnas import MLPNAS
from CONSTANTS import TOP_N


data = pd.read_excel('DATASETS/toBeUsed.xlsx')
x = data.drop(columns=['isBestSeller'], inplace=False).values
y = data['isBestSeller'].values

nas_object = MLPNAS(x, y)
data = nas_object.search()

get_top_n_architectures(TOP_N)
