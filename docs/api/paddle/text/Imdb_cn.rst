.. _cn_api_text_datasets_Imdb:

Imdb
-------------------------------

.. py:class:: paddle.text.datasets.Imdb()


该类是对`IMDB <https://www.imdb.com/interfaces/>`_ 测试数据集的实现。

参数
:::::::::
    - data_file(str) - 保存压缩数据的路径，如果参数:attr:`download`设置为True，
    可设置为None。默认为None。
    - mode(str) - 'train' 或'test' 模式。默认为'train'。
    - cutoff(int) - 构建词典的截止大小。默认为Default 150。
    - download(bool) - 如果:attr:`data_file`未设置，是否自动下载数据集。默认为True。

返回值
:::::::::
``Dataset``， IMDB数据集实例。

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


    imdb = Imdb(mode='train')

    for i in range(10):
        doc, label = imdb[i]
        doc = paddle.to_tensor(doc)
        label = paddle.to_tensor(label)

        model = SimpleNet()
        image, label = model(doc, label)
        print(doc.numpy().shape, label.numpy().shape)

