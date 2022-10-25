.. _cn_api_fluid_layers_argmin:

argmin
-------------------------------

.. py:function:: paddle.fluid.layers.argmin(x, axis=0)




**argmin**

该OP沿 ``axis`` 计算输入 ``x`` 的最小元素的索引。

参数
::::::::::::

    - **x** (Variable) - 输入的多维 ``Tensor``，支持的数据类型：float32、float64、int8、int16、int32、int64。
    - **axis** (int，可选) - 指定对输入Tensor进行运算的轴，``axis`` 的有效范围是[-R, R)，R是输入 ``x`` 的Rank， ``axis`` 为负时与 ``axis`` +R 等价。默认值为0。

返回
::::::::::::
 ``Tensor``，数据类型int64

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.argmin