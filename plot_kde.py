'''
#Draw the original figure

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5), dpi=80)   
p = plt.subplot(111)   

df1 = pd.read_excel(r"D:\workspace\Wifi\papers\tests&pictures\plot_kde.xlsx", "Sheet1")
df2 = pd.read_excel(r"D:\workspace\Wifi\papers\tests&pictures\plot_kde.xlsx", "Sheet2") 
type1 = p.scatter(df1.x, df1.y,s=40, c='red')   
type2 = p.scatter(df2.x, df2.y, s=40, c='green', marker='x')  

plt.xlabel('x')   
plt.ylabel('y')   
p.legend((type1, type2), ('shop 0', 'shop 1'), loc=4)   
plt.show()
'''

#Draw the KDE figure

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(color_codes=True, style="white")

#plt.plot([0, 15],[0,15])

df1 = pd.read_excel(r"D:\workspace\Wifi\papers\tests&pictures\plot_kde.xlsx", "Sheet1")
df2 = pd.read_excel(r"D:\workspace\Wifi\papers\tests&pictures\plot_kde.xlsx", "Sheet2")

sns.kdeplot(df1.x, df1.y, cmap="Reds", legend=True, shade_lowest=True)
sns.kdeplot(df2.x, df2.y, cmap="Greens", legend=True, shade_lowest=True)

#points = np.random.multivariate_normal([0, 0], [[1, 2], [2, 20]], size=1000)
#sns.kdeplot(points, cmap="Reds", legend=True, shade_lowest=True)

#points2 = np.random.multivariate_normal([0, 0], [[1, 2], [2, 20]], size=1000)
#sns.kdeplot(points2, cmap="Greens", legend=True, shade_lowest=True)

plt.show()

'''
#Draw the LR figure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True, style="white")

df1 = pd.read_excel(r"D:\workspace\Wifi\papers\tests&pictures\plot_kde.xlsx", "Sheet1")
#df2 = pd.read_excel(r"D:\workspace\Wifi\papers\tests&pictures\plot_kde.xlsx", "Sheet2")

sns.regplot(x=df1.x, y=df1.y, data=df1, color="red", label="shop 0")
sns.regplot(x=df2.x, y=df2.y, data=df2, color="g", marker="x", label="shop 1")

plt.show()
'''