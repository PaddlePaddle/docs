.. _cn_api_text_datasets_Movielens:

Movielens
-------------------------------

.. py:class:: paddle.text.datasets.Movielens()


    Implementation of `Movielens 1-M <https://grouplens.org/datasets/movielens/1m/>`_ dataset.

    参数
:::::::::
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train' or 'test' mode. Default 'train'.
        test_ratio(float): split ratio for test sample. Default 0.1.
        rand_seed(int): random seed. Default 0.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of Movielens 1-M dataset

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

            paddle.disable_static()

            movielens = Movielens(mode='train')

            for i in range(10):
                category, title, rating = movielens[i][-3:]
                category = paddle.to_tensor(category)
                title = paddle.to_tensor(title)
                rating = paddle.to_tensor(rating)

                model = SimpleNet()
                category, title, rating = model(category, title, rating)
                print(category.numpy().shape, title.numpy().shape, rating.numpy().shape)

    