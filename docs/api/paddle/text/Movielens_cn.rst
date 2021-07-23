.. _cn_api_text_datasets_Movielens:

Movielens
-------------------------------

.. py:class:: paddle.text.datasets.Movielens()


该类是对 `Movielens 1-M <https://grouplens.org/datasets/movielens/1m/>`_
测试数据集的实现。

参数
:::::::::
    - data_file（str）- 保存压缩数据的路径，如果参数:attr:`download`设置为True，
    可设置为None。默认为None。
    - mode（str）- 'train' 或 'test' 模式。默认为'train'。
    - test_ratio（float) - 为测试集划分的比例。默认为0.1。
    - rand_seed（int）- 随机数种子。默认为0。
    - download（bool）- 如果:attr:`data_file`未设置，是否自动下载数据集。默认为True。

返回值
:::::::::
    ``Dataset``，Movielens 1-M数据集实例。

代码示例
:::::::::

.. code-block:: python

    import paddle
    from paddle.text.datasets import Movielens

    class SimpleNet(paddle.nn.Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()

        def forward(self, category, title, rating):
            return paddle.sum(category), paddle.sum(title), paddle.sum(rating)


    movielens = Movielens(mode='train')

    for i in range(10):
        category, title, rating = movielens[i][-3:]
        category = paddle.to_tensor(category)
        title = paddle.to_tensor(title)
        rating = paddle.to_tensor(rating)

        model = SimpleNet()
        category, title, rating = model(category, title, rating)
        print(category.numpy().shape, title.numpy().shape, rating.numpy().shape)
