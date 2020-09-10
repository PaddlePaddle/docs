.. _cn_api_text_datasets_Imdb:

Imdb
-------------------------------

.. py:class:: paddle.text.datasets.Imdb()


    Implementation of `IMDB <https://www.imdb.com/interfaces/>`_ dataset.

    参数
:::::::::
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train' 'test' mode. Default 'train'.
        cutoff(int): cutoff number for building word dictionary. Default 150.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of IMDB dataset

    代码示例
:::::::::

        .. code-block:: python

            import paddle
            from paddle.text.datasets import Imdb

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()

                def forward(self, doc, label):
                    return paddle.sum(doc), label

            paddle.disable_static()

            imdb = Imdb(mode='train')

            for i in range(10):
                doc, label = imdb[i]
                doc = paddle.to_tensor(doc)
                label = paddle.to_tensor(label)

                model = SimpleNet()
                image, label = model(doc, label)
                print(doc.numpy().shape, label.numpy().shape)

    