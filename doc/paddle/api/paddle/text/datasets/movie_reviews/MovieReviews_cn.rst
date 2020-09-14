.. _cn_api_text_datasets_MovieReviews:

MovieReviews
-------------------------------

.. py:class:: paddle.text.datasets.MovieReviews()


该类是对`NLTK movie reviews <http://www.nltk.org/nltk_data/>`_ 测试数据集的实现。

参数
:::::::::
    - data_file（str）- 保存压缩数据的路径，如果参数:attr:`download`设置为True，
    可设置为None。默认为None。
    - mode（str）- 'train'或 'test' 模式。默认为'train'。
    - download（bool）- 如果:attr:`data_file`未设置，是否自动下载数据集。默认为True。

返回值
:::::::::
``Dataset``，NLTK movie reviews数据集实例。

代码示例
:::::::::

.. code-block:: python

    import paddle
    from paddle.text.datasets import MovieReviews

    class SimpleNet(paddle.nn.Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()

        def forward(self, word, category):
            return paddle.sum(word), category

    paddle.disable_static()

    movie_reviews = MovieReviews(mode='train')

    for i in range(10):
        word_list, category = movie_reviews[i]
        word_list = paddle.to_tensor(word_list)
        category = paddle.to_tensor(category)

        model = SimpleNet()
        word_list, category = model(word_list, category)
        print(word_list.numpy().shape, category.numpy())

