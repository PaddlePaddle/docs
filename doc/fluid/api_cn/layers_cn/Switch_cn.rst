.. _cn_api_fluid_layers_Switch:

Switch
-------------------------------

.. py:class:: paddle.fluid.layers.Switch (name=None)

Switch类实现的功能十分类似if-elif-else。它可以在学习率调度器(learning rate scheduler)中调整学习率。
::
  语义上，
      1. switch控制流挨个检查cases
      2. 各个case的条件是一个布尔值(boolean)，它是一个标量(scalar)变量
      3. 它将执行第一个匹配的case后面的分支，如果没有匹配的case，但若存在一个default case,则会执行default case后面的语句
      4. 一旦匹配了一个case,它降会执行这个case所对应的分支，且仅此分支。

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid

    lr = fluid.layers.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")
    zero_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=0.0)
    one_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=1.0)
    two_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=2.0)

    global_step = fluid.layers.autoincreased_step_counter(
           counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)

    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(global_step == zero_var):
            fluid.layers.assign(input=one_var, output=lr)
        with switch.default():
            fluid.layers.assign(input=two_var, output=lr)













