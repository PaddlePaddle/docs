.. _cn_api_fluid_layers_where:

where
-------------------------------

.. py:function:: paddle.fluid.layers.where(condition)




该 OP 计算输入元素中为 True 的元素在输入中的坐标（index）。

参数
::::::::::::

    - **condition** （Variable）– 输入秩至少为 1 的多维 Tensor，数据类型是 bool 类型。

返回
::::::::::::
输出 condition 元素为 True 的坐标（index），将所有的坐标（index）组成一个 2-D 的 Tensor。

返回类型
::::::::::::
Variable，数据类型是 int64。

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.where
