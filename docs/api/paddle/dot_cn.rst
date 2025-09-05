.. _cn_api_paddle_dot:

dot
-------------------------------

.. py:function:: paddle.dot(x, y, name=None)


计算向量的内积

.. note::

   支持 1 维和 2 维 Tensor。如果是 2 维 Tensor，矩阵的第一个维度是 batch_size，将会在多个样本上进行点积计算。

.. note::
    别名支持: 参数名 ``input`` 可替代 ``x``， ``tensor`` 可替代 ``y``。

参数
:::::::::

        - **x** （Tensor） - 1 维或 2 维 ``Tensor``。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 、 ``int64`` 、 ``complex64`` 或 ``complex128`` 。
          别名： ``input``
        - **y** （Tensor） - 1 维或 2 维 ``Tensor``。数据类型为 ``float32`` 、 ``float64`` 、 ``int32`` 、 ``int64`` 、 ``complex64`` 或 ``complex128`` 。
          别名： ``tensor``
        - **name** （str，可选） - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::

    - **out** （Tensor，可选） - 指定输出结果的 `Tensor`，默认值为 None。

返回
:::::::::
``Tensor``，数据类型与 ``x`` 相同。



代码示例
:::::::::

COPY-FROM: paddle.dot
