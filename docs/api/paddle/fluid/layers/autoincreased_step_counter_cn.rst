.. _cn_api_fluid_layers_autoincreased_step_counter:

autoincreased_step_counter
-------------------------------


.. py:function:: paddle.fluid.layers.autoincreased_step_counter(counter_name=None, begin=1, step=1)




创建一个自增变量，每个迭代累加一次，默认首次返回值为 1，默认累加步长为 1。

参数
::::::::::::

    - **counter_name** (str，可选) - 该计数器的名称，默认为 ``@STEP_COUNTER@`` 。
    - **begin** (int) - 该计数器返回的第一个值。
    - **step** (int) - 累加步长。

返回
::::::::::::
累加结果，数据类型为 int64

返回类型
::::::::::::
Variable

代码示例
::::::::::::

COPY-FROM: paddle.fluid.layers.autoincreased_step_counter