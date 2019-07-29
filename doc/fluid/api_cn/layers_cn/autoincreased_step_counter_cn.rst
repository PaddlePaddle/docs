.. _cn_api_fluid_layers_autoincreased_step_counter:

autoincreased_step_counter
-------------------------------

.. py:function:: paddle.fluid.layers.autoincreased_step_counter(counter_name=None, begin=1, step=1)

创建一个自增变量，每个mini-batch返回主函数运行次数，变量自动加1，默认初始值为1.

参数：
    - **counter_name** (str)-计数名称，默认为 ``@STEP_COUNTER@``
    - **begin** (int)-开始计数
    - **step** (int)-执行之间增加的步数

返回：全局运行步数

返回类型：变量（Variable）

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    global_step = fluid.layers.autoincreased_step_counter(
        counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)









