from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import korean_font

X, y, w = make_regression(
    n_samples=50, n_features=1, bias=100, noise=10, coef=True, random_state=0
)

xx = np.linspace(-3, 3, 100)
y0 = w * xx + 100
plt.plot(xx, y0, "r-")
plt.scatter(X, y, s=100)
plt.xlabel("x")
plt.ylabel("y")
plt.title("make_regression 예제")
plt.show()


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],  # 2 samples, 3 features
     [11, 12, 13]]
y = [0, 1]  # classes of each sample
clf.fit(X, y)

clf.predict(X)  # predict classes of the training data

clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data
from sklearn.preprocessing import StandardScaler
X = [[0, 15],
     [1, -10]]
# scale data according to computed scaling values
StandardScaler().fit(X).transform(X)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# load the iris dataset and split it into train and test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the whole pipeline
pipe.fit(X_train, y_train)


# we can now use it like any other estimator
accuracy_score(pipe.predict(X_test), y_test)


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
X.shape
y.shape(1000,)

lr = LinearRegression()

result = cross_validate(lr, X, y)  # defaults to 5-fold CV
result
result['test_score']  # r_squared score is high because dataset is easy

#
# Iris flower data set
# https://en.wikipedia.org/wiki/Iris_flower_data_set
#

from sklearn.datasets import load_iris

iris = load_iris()

# Pandas패키지로 데이터프레임을 만들어 일부 데이터를 살펴본다.
import pandas as pd
df = pd.DataFrame(iris.data, columns=iris.feature_names)
sy = pd.Series(iris.target, dtype="category")
sy = sy.cat.rename_categories(iris.target_names)
df['species'] = sy
df.tail()

# 각 특징값의 분포와 상관관계를 히스토그램과 스캐터플롯으로 나타내면 다음과 같다.
import seaborn as sns
sns.pairplot(df, hue="species")

import warnings
warnings.filterwarnings(action='ignore')

# 이 분포를 잘 살펴보면 꽃잎의 길이만으로도 세토사와 다른 종을 분류할 수 있다는 것을 알 수 있다.
def visualization():
    import matplotlib.pyplot as plt
    sns.distplot(df[df.species != "setosa"]["petal length (cm)"], hist=True, rug=True, label="setosa")
    sns.distplot(df[df.species == "setosa"]["petal length (cm)"], hist=True, rug=True, label="others")
    plt.legend()

visualization()

# 하지만 베르시칼라와 버지니카는 이 방법으로 완벽한 구분이 불가능하다.
def visualization():
    import matplotlib.pyplot as plt
    sns.distplot(df[df.species == "virginica"]["petal length (cm)"], hist=True, rug=True, label="virginica")
    sns.distplot(df[df.species == "versicolor"]["petal length (cm)"], hist=True, rug=True, label="versicolor")
    plt.legend()

visualization()
