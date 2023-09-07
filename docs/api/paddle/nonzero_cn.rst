.. _cn_api_paddle_nonzero:

nonzero
-------------------------------

.. py:function:: paddle.nonzero(x, as_tuple=False)




返回输入 ``x`` 中非零元素的坐标。如果输入 ``x`` 有 ``n`` 维，共包含 ``z`` 个非零元素，当 ``as_tuple = False`` 时，
返回结果是一个 ``shape`` 等于 ``[z x n]`` 的 ``Tensor``，第 ``i`` 行代表输入中第 ``i`` 个非零元素的坐标；当 ``as_tuple = True`` 时，
返回结果是由 ``n`` 个大小为 ``z`` 的 ``1-D Tensor`` 构成的元组，第 ``i`` 个 ``1-D Tensor`` 记录输入的非零元素在第 ``i`` 维的坐标。

参数
:::::::::

    - **x** （Tensor）– 输入的 Tensor。
    - **as_tuple** (bool，可选) - 返回格式。是否以 ``1-D Tensor`` 构成的元组格式返回。



返回
:::::::::
    - **Tensor or tuple(1-D Tensor)**，数据类型为 **INT64** 。



代码示例
:::::::::

COPY-FROM: paddle.nonzero
