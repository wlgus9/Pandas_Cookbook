##Standardization

import seaborn as sns
iris = sns.load_dataset('iris')

iris_X = iris.iloc[:, :-1]
iris_X.head()

#scale() : 평균 0, 표준편차 1
from sklearn.preprocessing import scale
iris_scaled = scale(iris_X)
iris_scaled[:5, :]

iris_scaled.mean(axis=0)

for scaled_mean in iris_scaled.mean(axis=0):
    print('{:10.9f}'.format(scaled_mean))

iris_scaled.std(axis=0)
