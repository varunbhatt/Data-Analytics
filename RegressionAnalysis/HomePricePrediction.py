import pandas as pd
import seaborn as sns 

import matplotlib.pyplot as plt

import numpy as np

housing_df = pd.read_csv('housing.csv')

plt.hist(housing_df['median_house_value'], bins=80)

plt.show()