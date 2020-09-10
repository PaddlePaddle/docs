.. _cn_api_text_datasets_UCIHousing:

UCIHousing
-------------------------------

.. py:class:: paddle.text.datasets.UCIHousing()


    Implementation of `UCI housing <https://archive.ics.uci.edu/ml/datasets/Housing>`_
    dataset

    参数
:::::::::
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train' or 'test' mode. Default 'train'.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of UCI housing dataset.

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

    