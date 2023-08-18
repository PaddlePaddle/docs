.. _cn_api_paddle_tensor_linalg_dot:

dot
-------------------------------

.. py:function:: paddle.dot(x, y, name=None)


计算向量的内积

.. note::

   支持 1 维和 2 维 Tensor。如果是 2 维 Tensor，矩阵的第一个维度是 batch_size，将会在多个样本上进行点积计算。

参数
:::::::::

        - **x** （Tensor）- 1 维或 2 维 ``Tensor``。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 、 ``int64`` 、 ``complex64`` 或 ``complex128`` 。
        - **y** （Tensor）- 1 维或 2 维 ``Tensor``。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 、 ``int64`` 、 ``complex64`` 或 ``complex128`` 。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
``Tensor``，数据类型与 ``x`` 相同。



代码示例
:::::::::

COPY-FROM: paddle.dot
