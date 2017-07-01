import pandas as pd
import numpy as np

dataframe = pd.read_csv('Skin_NonSkin.txt', delimiter='\t')
print(dataframe.head(5))

print('Labels')
label = dataframe['S']
print(label.head(5))
