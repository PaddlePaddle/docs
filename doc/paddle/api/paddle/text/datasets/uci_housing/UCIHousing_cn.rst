.. _cn_api_text_datasets_UCIHousing:

UCIHousing
-------------------------------

.. py:class:: paddle.text.datasets.UCIHousing()


该类是对`UCI housing <https://archive.ics.uci.edu/ml/datasets/Housing>`_
测试数据集的实现。

参数
:::::::::
    - data_file（str）- 保存数据的路径，如果参数:attr:`download`设置为True，
    可设置为None。默认为None。
    - mode（str）- 'train'或'test'模式。默认为'train'。
    - download（bool）- 如果:attr:`data_file`未设置，是否自动下载数据集。默认为True。

返回值
:::::::::
``Dataset``，UCI housing数据集实例。

代码示例
:::::::::
        
.. code-block:: python

    import paddle
    from paddle.text.datasets import UCIHousing

    class SimpleNet(paddle.nn.Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()

        def forward(self, feature, target):
            return paddle.sum(feature), target

    paddle.disable_static()

    uci_housing = UCIHousing(mode='train')

    for i in range(10):
        feature, target = uci_housing[i]
        feature = paddle.to_tensor(feature)
        target = paddle.to_tensor(target)

        model = SimpleNet()
        feature, target = model(feature, target)
        print(feature.numpy().shape, target.numpy())

