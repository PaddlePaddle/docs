.. _cn_api_vision_datasets_Cifar100:

Cifar100
-------------------------------

.. py:class:: paddle.vision.datasets.Cifar100()


    `Cifar-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ 数据集的实现，数据集包含100种类别.

参数
:::::::::
        - data_file (str) - 数据集文件路径，如果 ``download`` 参数设置为 ``True`` ， ``data_file`` 参数可以设置为 ``None`` 。默认值为 ``None`` ，默认存放在: ``~/.cache/paddle/dataset/cifar``
        - mode (str) - ``'train'`` 或 ``'test'`` 模式，默认为 ``'train'`` 。
        - transform (callable) - 图片数据的预处理，若为 ``None`` 即为不做预处理。默认值为 ``None``。
        - download (bool) - 当 ``data_file`` 是 ``None`` 时，该参数决定是否自动下载数据集文件。默认为 ``True`` 。
        - backend (str, optional) - 指定要返回的图像类型：PIL.Image或numpy.ndarray。必须是{'pil'，'cv2'}中的值。如果未设置此选项，将从paddle.vsion.get_image_backend获得这个值。 默认值： ``None`` 。
返回
:::::::::

				Cifar100数据集实例

代码示例
:::::::::

        .. code-block:: python

            import paddle
            import paddle.nn as nn
            from paddle.vision.datasets import Cifar100
            from paddle.vision.transforms import Normalize

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(3072, 10),
                        nn.Softmax())

                def forward(self, image, label):
                    image = paddle.reshape(image, (1, -1))
                    return self.fc(image), label


            normalize = Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5],
                                data_format='HWC')
            cifar100 = Cifar100(mode='train', transform=normalize)

            for i in range(10):
                image, label = cifar100[i]
                image = paddle.to_tensor(image)
                label = paddle.to_tensor(label)

                model = SimpleNet()
                image, label = model(image, label)
                print(image.numpy().shape, label.numpy().shape)


