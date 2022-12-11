.. _cn_api_fluid_layers_less_than:

less_than
-------------------------------

.. py:function:: paddle.fluid.layers.less_than(x, y, force_cpu=None, cond=None, name=None)





该OP逐元素地返回 :math:`x < y` 的逻辑值，使用重载算子 `<` 可以有相同的计算函数效果


参数
::::::::::::

    - **x** (Variable) - 进行比较的第一个输入，是一个多维的LoDTensor/Tensor，数据类型可以是float32，float64，int32，int64。
    - **y** (Variable) - 进行比较的第二个输入，是一个多维的LoDTensor/Tensor，数据类型可以是float32，float64，int32，int64。
    - **force_cpu** (bool) – 如果为True则强制将输出变量写入CPU内存中，否则将其写入目前所在的运算设备上。默认值为False。注意：该属性已弃用，其值始终是False。
    - **cond** (Variable，可选) – 指定算子输出结果的LoDTensor/Tensor，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。


返回
::::::::::::
输出结果的LoDTensor/Tensor，数据的shape和输入x一致。

返回类型
::::::::::::
 Variable，数据类型为bool。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.less_than