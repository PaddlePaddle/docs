.. _cn_api_vision_datasets_VOC2012:

VOC2012
-------------------------------

.. py:class:: paddle.vision.datasets.VOC2012()


    Implementation of `VOC2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>`_ dataset

    参数
:::::::::
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train', 'valid' or 'test' mode. Default 'train'.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    代码示例
:::::::::

        .. code-block:: python

            import paddle
            from paddle.vision.datasets import VOC2012

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()

                def forward(self, image, label):
                    return paddle.sum(image), label

            paddle.disable_static()

            voc2012 = VOC2012(mode='train')

            for i in range(10):
                image, label= voc2012[i]
                image = paddle.cast(paddle.to_tensor(image), 'float32')
                label = paddle.to_tensor(label)

                model = SimpleNet()
                image, label= model(image, label)
                print(image.numpy().shape, label.numpy().shape)

    