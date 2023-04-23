.. _cn_api_tensor_floor_divide:

floor_divide
-------------------------------

.. py:function:: paddle.floor_divide(x, y, name=None)

逐元素整除算子，将输入 ``x`` 与输入 ``y`` 逐元素整除（商被朝 0 方向舍入到最接近的整数值），并将各个位置的输出元素保存到返回结果中。

.. note::
    输入 ``x`` 与输入 ``y`` 必须和广播为相同形状，关于广播规则，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

等式为：

.. math::
        Out = trunc(X / Y)

- :math:`X`：多维 Tensor。
- :math:`Y`：多维 Tensor。

注意
:::::::::
`floor_divide`的名称可能带来误导，因为商被向零而非向负无穷舍入。

参数
:::::::::
        - **x** (Tensor) - 多维 Tensor。数据类型为 int32 或 int64。
        - **y** (Tensor) - 多维 Tensor。数据类型为 int32 或 int64。
        - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
:::::::::
多维 Tensor，数据类型与 ``x`` 相同，维度为广播后的形状。


代码示例
:::::::::

COPY-FROM: paddle.floor_divide
