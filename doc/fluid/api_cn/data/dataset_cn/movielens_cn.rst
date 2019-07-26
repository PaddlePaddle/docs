.. _cn_api_paddle_dataset_movielens:

movielens
-------------------------------


Movielens 1-M数据集。

Movielens 1-M数据集是由GroupLens Research采集的6000个用户对4000个电影的的100万个评级。 该模块将从 http://files.grouplens.org/datasets/movielens/ml-1m.zip 下载Movielens 1-M数据集，并将训练集和测试集解析为paddle reader creator。


.. py:function:: paddle.dataset.movielens.get_movie_title_dict()

获取电影标题词典。

.. py:function:: paddle.dataset.movielens.max_movie_id()

获取电影ID的最大值。


.. py:function:: paddle.dataset.movielens.max_user_id()

获取用户ID的最大值。


.. py:function:: paddle.dataset.movielens.max_job_id()

获取职业ID的最大值。


.. py:function:: paddle.dataset.movielens.movie_categories()

获取电影类别词典。

.. py:function:: paddle.dataset.movielens.user_info()

获取用户信息词典。

.. py:function:: paddle.dataset.movielens.movie_info()

获取电影信息词典。

.. py:function:: paddle.dataset.movielens.convert(path)

将数据集转换为recordio格式。

.. py:class:: paddle.dataset.movielens.MovieInfo(index, categories, title)

电影ID，标题和类别信息存储在MovieInfo中。


.. py:class:: paddle.dataset.movielens.UserInfo(index, gender, age, job_id)

用户ID，性别，年龄和工作信息存储在UserInfo中。



