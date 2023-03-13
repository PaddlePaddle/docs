.. _cn_api_fluid_layers_less_than:

less_than
-------------------------------

.. py:function:: paddle.fluid.layers.less_than(x, y, force_cpu=None, cond=None, name=None)





该 OP 逐元素地返回 :math:`x < y` 的逻辑值，使用重载算子 `<` 可以有相同的计算函数效果


参数
::::::::::::

    - **x** (Variable) - 进行比较的第一个输入，是一个多维的 LoDTensor/Tensor，数据类型可以是 float32，float64，int32，int64。
    - **y** (Variable) - 进行比较的第二个输入，是一个多维的 LoDTensor/Tensor，数据类型可以是 float32，float64，int32，int64。
    - **force_cpu** (bool) – 如果为 True 则强制将输出变量写入 CPU 内存中，否则将其写入目前所在的运算设备上。默认值为 False。注意：该属性已弃用，其值始终是 False。
    - **cond** (Variable，可选) – 指定算子输出结果的 LoDTensor/Tensor，可以是程序中已经创建的任何 Variable。默认值为 None，此时将创建新的 Variable 来保存输出结果。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
输出结果的 LoDTensor/Tensor，数据的 shape 和输入 x 一致。

返回类型
::::::::::::
 Variable，数据类型为 bool。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.less_than
