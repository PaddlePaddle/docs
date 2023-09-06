.. _cn_api_vision_transforms_Compose:

Compose
-------------------------------

.. py:class:: paddle.vision.transforms.Compose(transforms)

将用于数据集预处理的接口以列表的方式进行组合。

参数
:::::::::

    - **transforms** (list|tuple) - 用于组合的数据预处理接口实例列表。

返回
:::::::::

    一个可调用的 Compose 对象，它将依次调用每个给定的 :attr:`transforms`。

代码示例
:::::::::

COPY-FROM: paddle.vision.transforms.Compose
