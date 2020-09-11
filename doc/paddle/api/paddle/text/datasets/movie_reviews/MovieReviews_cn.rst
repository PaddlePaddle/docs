.. _cn_api_text_datasets_MovieReviews:

MovieReviews
-------------------------------

.. py:class:: paddle.text.datasets.MovieReviews()


    Implementation of `NLTK movie reviews <http://www.nltk.org/nltk_data/>`_ dataset.

    参数
:::::::::
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train' 'test' mode. Default 'train'.
        download(bool): whether auto download cifar dataset if
            :attr:`data_file` unset. Default True.

    Returns:
        Dataset: instance of movie reviews dataset

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

    