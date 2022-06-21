.. _cn_api_fluid_layers_where:

where
-------------------------------

.. py:function:: paddle.fluid.layers.where(condition)




该OP计算输入元素中为True的元素在输入中的坐标（index）。
        
参数
::::::::::::

    - **condition** （Variable）– 输入秩至少为1的多维Tensor，数据类型是bool类型。

返回
::::::::::::
输出condition元素为True的坐标（index），将所有的坐标（index）组成一个2-D的Tensor。

返回类型
::::::::::::
Variable，数据类型是int64。
     
代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.where