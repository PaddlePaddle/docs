.. _cn_api_fluid_layers_Switch:

Switch
-------------------------------


.. py:class:: paddle.fluid.layers.Switch (name=None)




该类用于实现 Switch 分支控制功能。Switch 分支包含多个 case 分支和一个 default 分支，Switch 控制流会依次检查各 case 分支条件是否满足，并仅执行第一个满足条件的 case 分支后面的语句。若不存在满足条件的 case 分支，则仅执行 default 分支后面的语句。

.. note::
    如果参数 ``cond`` 的形状为[1]，强烈建议您使用新的 OP :ref:`cn_api_fluid_layers_case` 而不是 ``Switch``。
    OP :ref:`cn_api_fluid_layers_case` 的使用方式更简单，并且调用该 OP 所用的代码更少且功能与 ``Switch`` 一样。

成员函数：
    - **case(cond)** - Switch 的 case 分支，其参数 cond 为 bool 型的标量 Variable。只有当前 case 分支的 cond 为 True，且之前的 case 分支的 cond 均为 False，该 case 分支后的语句才会执行，且不再执行之后的 case 后的语句。
    - **default()** - Switch 的 default 分支。当所有 case 分支的 cond 均为 False 时，执行 default 分支后的语句。

注意：case 和 default 函数只能用于 Switch 的 scope 内部，示例如下：

..  code-block:: python

    with fluid.layers.Switch() as switch:
        with switch.case(cond1):
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
        with switch.case(cond2):
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2)
        with switch.default():
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)

参数
::::::::::::

    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name`，一般无需设置，默认值为 None。

代码示例
::::::::::::

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

    # 将参数中的 begin 设为非 0 值，则进入 Switch 的 default 分支，输出数组中的数字将为 2
    global_step = fluid.layers.autoincreased_step_counter(counter_name='@LR_DECAY_COUNTER@', begin=0, step=1)

    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(global_step == zero_var):
            fluid.layers.assign(input=one_var, output=lr)
        with switch.default():
            fluid.layers.assign(input=two_var, output=lr)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    res = exe.run(fluid.default_main_program(), feed={}, fetch_list=[lr])
    print(res) # [array([1.], dtype=float32)]
