.. _cn_api_text_datasets_Imikolov:

Imikolov
-------------------------------

.. py:class:: paddle.text.datasets.Imikolov()


该类是对imikolov测试数据集的实现。

参数
:::::::::
    - data_file（str）- 保存数据的路径，如果参数:attr:`download`设置为True，
    可设置为None。默认为None。
    - data_type（str）- 'NGRAM'或'SEQ'。默认为'NGRAM'。
    - window_size（int) - 'NGRAM'数据滑动窗口的大小。默认为-1。
    - mode（str）- 'train' 'test' mode. Default 'train'.
    - min_word_freq（int）- 构建词典的最小词频。默认为50。
    - download（bool）- 如果:attr:`data_file`未设置，是否自动下载数据集。默认为True。

返回值
:::::::::
``Dataset``，imikolov数据集实例。

代码示例
:::::::::

.. code-block:: python

    import paddle
    from paddle.text.datasets import Imikolov

    class SimpleNet(paddle.nn.Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()

        def forward(self, src, trg):
            return paddle.sum(src), paddle.sum(trg)

    paddle.disable_static()

    imikolov = Imikolov(mode='train', data_type='SEQ', window_size=2)

    for i in range(10):
        src, trg = imikolov[i]
        src = paddle.to_tensor(src)
        trg = paddle.to_tensor(trg)

        model = SimpleNet()
        src, trg = model(src, trg)
        print(src.numpy().shape, trg.numpy().shape)

