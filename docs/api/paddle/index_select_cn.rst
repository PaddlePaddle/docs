.. _cn_api_paddle_index_select:

index_select
-------------------------------

.. py:function:: paddle.index_select(x, index, axis=0, name=None, *, out=None)



沿着指定轴  ``axis``  对输入  ``x``  进行索引，取  ``index``  中指定的相应项，创建并返回到一个新的 Tensor。这里  ``index``  是一个  ``1-D``  Tensor。除  ``axis``  轴外，返回的 Tensor 其余维度大小和输入  ``x``  相等，  ``axis``  维度的大小等于  ``index``  的大小。

.. note::
    别名支持: 参数名  ``input``  可替代  ``x`` ，  ``dim``  可替代  ``axis`` ；
    参数顺序支持: 支持 PyTorch 参数顺序  ``(input, dim, index)``  作为位置参数传入，可转换为 Paddle 顺序  ``(x, index, axis)`` ；
    如  ``(input=x, dim=1, index=idx)``  等价于  ``(x=x, index=idx, axis=1)`` ，  ``(x, 1, idx)``  等价于  ``(x, idx, axis=1)`` 。

参数
:::::::::

    - **x** （Tensor） - 输入 Tensor。  ``x``  的数据类型可以是 float16，float32，float64，int32，int64，complex64，complex128。
      别名：  ``input`` 
    - **index** （Tensor） - 包含索引下标的 1-D Tensor。
    - **axis** （int，可选） - 索引轴，若未指定，则默认选取第 0 维。
      别名：  ``dim`` 
    - **name** （str，可选） - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

关键字参数
:::::::::

    - **out** （Tensor，可选） - 指定输出结果的 `Tensor`，默认值为 None。

返回
:::::::::

Tensor，返回一个数据类型同输入的 Tensor。


代码示例
::::::::::::

COPY-FROM: paddle.index_select
