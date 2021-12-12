#유타 주의 알타 스키 리조트의 적설량 데이터를 읽고
#각 시즌에 얼마만큼의 눈이 왔는지 시각화

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

alta = pd.read_csv("csv/alta-noaa-1980-2019.csv")
alta

data = (alta
       .assign(DATE = pd.to_datetime(alta.DATE))
       .set_index('DATE')
       .loc['2018-09':'2019-08']
       .SNWD
       )
data

blue = '#99ddee'
white = '#ffffff'
fig, ax = plt.subplots(figsize = (12, 4), linewidth = 5, facecolor = blue)
ax.set_facecolor(blue)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis = 'x', colors = white)
ax.tick_params(axis = 'y', colors = white)
ax.set_ylabel('Snow Depth (in)', color = white)
ax.set_title('2018-2019', color = white, fontweight = 'bold')
ax.fill_between(data.index, data, color = white)
fig.savefig('c13-alta.png', dpi = 300, facecolor = blue)
fig
