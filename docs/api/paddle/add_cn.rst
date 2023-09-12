.. _cn_api_paddle_add:

add
-------------------------------

.. py:function:: paddle.add(x, y, name=None)



逐元素相加算子，输入 ``x`` 与输入 ``y`` 逐元素相加，并将各个位置的输出元素保存到返回结果中。

.. note::
输入 ``x`` 与输入 ``y`` 必须和广播为相同形状，关于广播规则，请参见 `Tensor 介绍`_ .

    .. _Tensor 介绍: ../../guides/beginner/tensor_cn.html#id7

等式为：

.. math::
        Out = X + Y

- :math:`X`：多维 Tensor。
- :math:`Y`：多维 Tensor。

如下两种情况使用该算子：
第一种情况：
1. ``X`` 与 ``Y`` 的形状一样。
2. ``Y`` 的形状是 ``X`` 的一部分连续的形状。
第二种情况：
1. 广播 ``Y`` 使其形状与 ``X`` 相同，其中 ``axis`` 是把 ``Y`` 广播到 ``X`` 的索引起始数。
2. 如果 ``axis`` 是默认值-1，则 :math:`axis=rank(X)−rank(Y)`。
3. ``Y`` 的最后一维形状为 1 时，该维度将会被忽略，比如 shape(Y) = (2, 1) => (2)。


参数
:::::::::
    - **x** (Tensor) - 输入的 Tensor，数据类型为：float32、float64、int32、int64。
    - **y** (Tensor) - 输入的 Tensor，数据类型为：float32、float64、int32、int64。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

返回
:::::::::
多维 Tensor，数据类型与 ``x`` 相同，维度为广播后的形状。


代码示例
:::::::::

COPY-FROM: paddle.add
