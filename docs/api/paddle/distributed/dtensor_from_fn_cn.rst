.. _cn_api_paddle_distributed_dtensor_from_fn:

dtensor_from_fn
-------------------------------

.. py:class:: paddle.distributed.dtensor_from_fn(fn, dist_attr, *args, **kwargs)

通过一个 paddle API (一般是 Tensor 创建类的 API )结合分布式属性 dist_attr 创建一个带分布式属性的 Tensor。

参数
:::::::::

    - **fn**  - paddle 公开的可创建 Tensor 的 API。例如: :ref:`cn_api_paddle_empty` 、 :ref:`cn_api_paddle_ones` 、:ref:`cn_api_paddle_zeros` 等 paddle API。
    - **dist_attr** (paddle.distributed.DistAttr) - 描述 Tensor 在 ProcessMesh 上的分布或切片方式。
    - ***args**  - fn 函数的输入参数( Tuple 形式)
    - ****kwargs**  - fn 函数的输入参数( Dict 形式)


返回
:::::::::
带有分布式信息的 Tensor



**代码示例**

COPY-FROM: paddle.distributed.dtensor_from_fn
