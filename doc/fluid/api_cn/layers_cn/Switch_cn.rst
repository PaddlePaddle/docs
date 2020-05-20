.. _cn_api_fluid_layers_Switch:

Switch
-------------------------------


.. py:class:: paddle.fluid.layers.Switch (name=None)

:api_attr: 声明式编程模式（静态图)



该类用于实现Switch分支控制功能。Switch分支包含多个case分支和一个default分支，Switch控制流会依次检查各case分支条件是否满足，并仅执行第一个满足条件的case分支后面的语句。若不存在满足条件的case分支，则仅执行default分支后面的语句。

.. note::
    如果参数 ``cond`` 的形状为[1]，强烈建议您使用新的OP :ref:`cn_api_fluid_layers_case` 而不是 ``Switch``。
    OP :ref:`cn_api_fluid_layers_case` 的使用方式更简单，并且调用该OP所用的代码更少且功能与 ``Switch`` 一样。

成员函数：
    - **case(cond)** - Switch的case分支，其参数cond为bool型的标量Variable。只有当前case分支的cond为True，且之前的case分支的cond均为False，该case分支后的语句才会执行，且不再执行之后的case后的语句。
    - **default()** - Switch的default分支。当所有case分支的cond均为False时，执行default分支后的语句。

注意：case和default函数只能用于Switch的scope内部，示例如下：

..  code-block:: python

    with fluid.layers.Switch() as switch:
        with switch.case(cond1):
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
        with switch.case(cond2):
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2)
        with switch.default():
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)

参数：
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

**代码示例**

..  code-block:: python

    with fluid.layers.Switch() as switch:
        with switch.case(cond1):
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=1)
        with switch.case(cond2):
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=2)
        with switch.default():
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)

