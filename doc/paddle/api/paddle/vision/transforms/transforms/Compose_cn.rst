.. _cn_api_vision_transforms_Compose:

Compose
-------------------------------

.. py:class:: paddle.vision.transforms.Compose(transforms)

将用于数据集预处理的接口以列表的方式进行组合。

参数
:::::::::

    - transforms (list) - 用于组合的数据预处理接口实例列表。

返回
:::::::::

    一个可调用的Compose对象，它将依次调用每个给定的 :attr:`transforms`。

代码示例
:::::::::
    
.. code-block:: python

    from paddle.vision.datasets import Flowers
    from paddle.vision.transforms import Compose, ColorJitter, Resize

    transform = Compose([ColorJitter(), Resize(size=608)])
    flowers = Flowers(mode='test', transform=transform)

    for i in range(10):
        sample = flowers[i]
        print(sample[0].shape, sample[1])

    