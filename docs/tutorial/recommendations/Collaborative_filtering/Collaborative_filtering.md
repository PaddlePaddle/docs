# 使用协同过滤实现电影推荐

**作者：** [HUANGCHENGAI](https://github.com/HUANGCHENGAI) <br>
**日期：** 2021.03 <br>
**摘要：** 本案例使用飞桨框架实现推荐电影的协同过滤算法。

## 一、介绍

此示例演示使用[Movielens 数据集](https://www.kaggle.com/c/movielens-100k)基于PaddlePaddle2.0向用户推荐电影的协作过滤算法。MovieLens 评级数据集列出了一组用户对一组电影的评分。目标是能够预测用户尚未观看的电影的收视率。然后，可以向用户推荐预测收视率最高的电影。

模型中的步骤如下：

    1.通过嵌入矩阵将用户 ID 映射到"用户向量"

    2.通过嵌入矩阵将电影 ID 映射到"电影载体"

    3.计算用户矢量和电影矢量之间的点产品，以获得用户和电影之间的匹配分数（预测评级）。

    4.使用所有已知的用户电影对通过梯度下降训练嵌入。


引用：

+ [Item-based collaborative filtering recommendation algorithms](https://dl.acm.org/doi/pdf/10.1145/371920.372071)

+ [Neural Collaborative Filtering](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569)

## 二、 环境设置

本教程基于Paddle 2.0 编写，如果您的环境不是本版本，请先参考官网[安装](https://www.paddlepaddle.org.cn/install/quick) Paddle 2.0 。


```python
import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset

print(paddle.__version__)
```

    2.0.1


## 三、数据集

这个数据集（ml-latest-small）描述了[MovieLens](http://movielens.org)的五星评级和自由文本标记活动。它包含100836个收视率和3683个标签应用程序，涵盖9742部电影。这些数据由610名用户在1996年3月29日至2018年9月24日期间创建。

该数据集于2018年9月26日生成，用户是随机选择的。所有选定的用户都对至少20部电影进行了评分。不包括人口统计信息。每个用户都由一个id表示，不提供其他信息。数据包含在文件中`links.csv`, `movies.csv`, `ratings.csv`以及`tags.csv`。

**用户ID**

MovieLens的用户是随机选择的

**电影ID**

数据集中只包含至少具有一个分级或标记的电影，这些电影id与MovieLens网站上使用的一致.。

分级数据文件结构(ratings.csv)

所有评级都包含在文件中`ratings.csv`. 文件头行后的每一行代表一个用户对一部电影的一个分级，格式如下：
userId，movieId，rating，timestamp


**标记数据文件结构(tags.csv)**

文件中包含所有标记`tags.csv`. 文件头行后的每一行代表一个用户应用于一部电影的一个标记，格式如下：
userId，movieId，tag，timestamp


**电影数据文件结构(movies.csv)**

格式如下：
电影ID、片名、类型

**链接数据文件结构(links.csv)**

格式如下：
电影ID，imdbId，tmdbId


```python
!unzip data/data71839/ml-latest-small.zip
```

    Archive:  data/data71839/ml-latest-small.zip
       creating: ml-latest-small/
      inflating: ml-latest-small/links.csv
      inflating: ml-latest-small/tags.csv
      inflating: ml-latest-small/ratings.csv
      inflating: ml-latest-small/README.txt
      inflating: ml-latest-small/movies.csv


### 3.1 数据处理

执行一些预处理，将用户和电影编码为整数指数


```python
df = pd.read_csv('ml-latest-small/ratings.csv')
user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
df["rating"] = df["rating"].values.astype(np.float32)
# 最小和最大额定值将在以后用于标准化额定值
min_rating = min(df["rating"])
max_rating = max(df["rating"])

print(
    "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_movies, min_rating, max_rating
    )
)
```

    Number of users: 610, Number of Movies: 9724, Min rating: 0.5, Max rating: 5.0


### 3.2 准备训练和验证数据


```python
df = df.sample(frac=1, random_state=42)
x = df[["user", "movie"]].values
# 规范化0和1之间的目标。使训练更容易。
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# 假设对90%的数据进行训练，对10%的数据进行验证。
train_indices = int(0.9 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)
y_train = y_train[: ,np.newaxis]
y_val = y_val[: ,np.newaxis]
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)

# 自定义数据集
#映射式(map-style)数据集需要继承paddle.io.Dataset
class SelfDefinedDataset(Dataset):
    def __init__(self, data_x, data_y, mode = 'train'):
        super(SelfDefinedDataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'predict':
           return self.data_x[idx]
        else:
           return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return len(self.data_x)

traindataset = SelfDefinedDataset(x_train, y_train)
for data, label in traindataset:
    print(data.shape, label.shape)
    print(data, label)
    break
train_loader = paddle.io.DataLoader(traindataset, batch_size = 128, shuffle = True)
for batch_id, data in enumerate(train_loader()):
    x_data = data[0]
    y_data = data[1]

    print(x_data.shape)
    print(y_data.shape)
    break

testdataset = SelfDefinedDataset(x_val, y_val)
test_loader = paddle.io.DataLoader(testdataset, batch_size = 128, shuffle = True)
for batch_id, data in enumerate(test_loader()):
    x_data = data[0]
    y_data = data[1]

    print(x_data.shape)
    print(y_data.shape)
    break

```

    (2,) (1,)
    [ 431 4730] [0.8888889]
    [128, 2]
    [128, 1]
    [128, 2]
    [128, 1]


## 四、模型组网

将用户和电影嵌入到 50 维向量中。

该模型计算用户和电影嵌入之间的匹配分数，并添加每部电影和每个用户的偏差。比赛分数通过 sigmoid 缩放到间隔[0, 1]。


```python
EMBEDDING_SIZE = 50

class RecommenderNet(nn.Layer):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        weight_attr_user = paddle.ParamAttr(
            regularizer = paddle.regularizer.L2Decay(1e-6),
            initializer = nn.initializer.KaimingNormal()
            )
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_size,
            weight_attr=weight_attr_user
        )
        self.user_bias = nn.Embedding(num_users, 1)
        weight_attr_movie = paddle.ParamAttr(
            regularizer = paddle.regularizer.L2Decay(1e-6),
            initializer = nn.initializer.KaimingNormal()
            )
        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_size,
            weight_attr=weight_attr_movie
        )
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = paddle.dot(user_vector, movie_vector)
        x = dot_user_movie + user_bias + movie_bias
        x = nn.functional.sigmoid(x)

        return x
```

## 五、模型训练

后台可通过VisualDl观察Loss曲线。


```python
model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
```


```python
model = paddle.Model(model)

optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0003)
loss = nn.BCELoss()
metric = paddle.metric.Accuracy()

# 设置visualdl路径
log_dir = './visualdl'
callback = paddle.callbacks.VisualDL(log_dir=log_dir)

model.prepare(optimizer, loss, metric)
model.fit(train_loader, epochs=5, save_dir='./checkpoints', verbose=1, callbacks=callback)

```

    The loss value printed in the log is the current step, and the metric is the average value of previous step.
    Epoch 1/5
    step  70/709 [=>............................] - loss: 0.6928 - acc: 0.8712 - ETA: 2s - 3ms/st

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.INT64, but right dtype is VarType.FP32, the right dtype will convert to VarType.INT64
      format(lhs_dtype, rhs_dtype, lhs_dtype))


    step 120/709 [====>.........................] - loss: 0.6908 - acc: 0.8718 - ETA: 1s - 3ms/stepstep 709/709 [==============================] - loss: 0.6730 - acc: 0.8687 - 3ms/step
    save checkpoint at /home/aistudio/checkpoints/0
    Epoch 2/5
    step 709/709 [==============================] - loss: 0.6548 - acc: 0.8687 - 3ms/step
    save checkpoint at /home/aistudio/checkpoints/1
    Epoch 3/5
    step 709/709 [==============================] - loss: 0.6267 - acc: 0.8687 - 3ms/step
    save checkpoint at /home/aistudio/checkpoints/2
    Epoch 4/5
    step 709/709 [==============================] - loss: 0.6012 - acc: 0.8687 - 3ms/step
    save checkpoint at /home/aistudio/checkpoints/3
    Epoch 5/5
    step 709/709 [==============================] - loss: 0.6231 - acc: 0.8687 - 3ms/step
    save checkpoint at /home/aistudio/checkpoints/4
    save checkpoint at /home/aistudio/checkpoints/final


## 六、模型评估


```python
model.evaluate(test_loader, batch_size=64, verbose=1)
```

    Eval begin...
    The loss value printed in the log is the current batch, and the metric is the average value of previous step.
    step 79/79 [==============================] - loss: 0.5982 - acc: 0.8713 - 3ms/step
    Eval samples: 10084


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/math_op_patch.py:238: UserWarning: The dtype of left and right variables are not the same, left dtype is VarType.INT64, but right dtype is VarType.FP32, the right dtype will convert to VarType.INT64
      format(lhs_dtype, rhs_dtype, lhs_dtype))





    {'loss': [0.5982282], 'acc': 0.8712812376041253}



## 七、模型预测


```python
movie_df = pd.read_csv('ml-latest-small/movies.csv')

# 获取一个用户，查看他的推荐电影
user_id = df.userId.sample(1).iloc[0]
movies_watched_by_user = df[df.userId == user_id]
movies_not_watched = movie_df[
    ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
]["movieId"]
movies_not_watched = list(
    set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
)
movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
user_encoder = user2user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
)
testdataset = SelfDefinedDataset(user_movie_array, user_movie_array, mode = 'predict')
test_loader = paddle.io.DataLoader(testdataset, batch_size = 9703, shuffle = False, return_list=True,)

ratings = model.predict(test_loader)
ratings = np.array(ratings)
ratings = np.squeeze(ratings, 0)
ratings = np.squeeze(ratings, 2)
ratings = np.squeeze(ratings, 0)
top_ratings_indices = ratings.argsort()[::-1][0:10]

print(top_ratings_indices)
recommended_movie_ids = [
    movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
]

print("用户的ID为: {}".format(user_id))
print("====" * 8)
print("用户评分较高的电影：")
print("----" * 8)
top_movies_user = (
    movies_watched_by_user.sort_values(by="rating", ascending=False)
    .head(5)
    .movieId.values
)
movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ":", row.genres)

print("----" * 8)
print("为用户推荐的10部电影：")
print("----" * 8)
recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.title, ":", row.genres)
```

    Predict begin...
    step 1/1 [==============================] - 17ms/step
    Predict samples: 9492
    [ 280  261  318   43  230  472 2393 8253  964 1874]
    用户的ID为: 594
    ================================
    用户评分较高的电影：
    --------------------------------
    Demolition Man (1993) : Action|Adventure|Sci-Fi
    Executive Decision (1996) : Action|Adventure|Thriller
    Matrix, The (1999) : Action|Sci-Fi|Thriller
    Bruce Almighty (2003) : Comedy|Drama|Fantasy|Romance
    Chasing Liberty (2004) : Comedy|Romance
    --------------------------------
    为用户推荐的10部电影：
    --------------------------------
    Usual Suspects, The (1995) : Crime|Mystery|Thriller
    Star Wars: Episode IV - A New Hope (1977) : Action|Adventure|Sci-Fi
    Pulp Fiction (1994) : Comedy|Crime|Drama|Thriller
    Shawshank Redemption, The (1994) : Crime|Drama
    Forrest Gump (1994) : Comedy|Drama|Romance|War
    Schindler's List (1993) : Drama|War
    Star Wars: Episode V - The Empire Strikes Back (1980) : Action|Adventure|Sci-Fi
    American History X (1998) : Crime|Drama
    Fight Club (1999) : Action|Crime|Drama|Thriller
    Dark Knight, The (2008) : Action|Crime|Drama|IMAX


请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions.
