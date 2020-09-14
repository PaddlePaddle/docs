.. _cn_api_vision_datasets_VOC2012:

VOC2012
-------------------------------

.. py:class:: paddle.vision.datasets.VOC2012()


    `VOC2012 <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/>`_ 数据集

参数
:::::::::
        - data_file (str) - 数据集文件路径，如果 ``download`` 设置为 ``True`` ，此参数可以设置为None。默认值为None。
        - label_file (str) - 标签文件路径，如果 ``download`` 设置为 ``True`` ，此参数可以设置为None。默认值为None。
        - setid_file (str) - 子数据集下标划分文件路径，如果 ``download`` 设置为 ``True`` ，此参数可以设置为None。默认值为None。
        - mode (str) - ``'train'`` 或 ``'test'`` 模式，默认为 ``'train'`` 。
        - transform (callable) - 作用于图片数据的transform，若未 ``None`` 即为无transform。
        - download (bool) - 是否自定下载数据集文件。默认为 ``True`` 。

返回
:::::::::

				VOC2012数据集实例

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

    
