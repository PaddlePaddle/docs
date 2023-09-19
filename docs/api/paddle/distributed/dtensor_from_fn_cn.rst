.. _cn_api_distributed_dtensor_from_fn:

dtensor_from_fn
-------------------------------

.. py:class:: paddle.distributed.dtensor_from_fn(fn, dist_attr, *args, **kwargs)

通过一个paddle API(一般是Tensor创建类的API)结合分布式属性dist_attr创建一个带有分布式属性的Tensor。

参数
:::::::::

    - **fn**  - paddle公开的可创建Tensor的API。例如:`paddle.empty`_、`paddle.ones`_、`paddle.zeros`_等paddle API
    .. _paddle.empty:https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/creation.py
    .. _paddle.ones:https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/creation.py
    .. _paddle.zeros:https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/creation.py
    - **dist_attr** (paddle.distributed.DistAttr) - 描述 Tensor 在 ProcessMesh 上的分布或切片方式。
    - ***args**  - fn函数的输入参数(Tuple形式)
    - ****kwargs**  - fn函数的输入参数(Dict形式)
    

返回
:::::::::
带有分布式信息的 Tensor



**代码示例**

COPY-FROM: paddle.distributed.dtensor_from_fn
