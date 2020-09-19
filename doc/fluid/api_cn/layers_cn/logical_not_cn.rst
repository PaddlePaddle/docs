.. _cn_api_fluid_layers_logical_not:

logical_not
-------------------------------

.. py:function:: paddle.logical_not(x, out=None, name=None)

:alias_main: paddle.logical_not
:alias: paddle.logical_not, paddle.tensor.logical_not, paddle.tensor.logic.logical_not
:old_api: paddle.fluid.layers.logical_not



该OP逐元素的对 ``X``  Variable进行逻辑非运算

.. math::
        Out = !X

参数：
        - **x** （Variable）- 逻辑非运算的输入，是一个 Variable，数据类型只能是bool。
        - **out** （Variable，可选）- 指定算子输出结果的 Variable，可以是程序中已经创建的任何 Variable。默认值为None，此时将创建新的Variable来保存输出结果。
        - **name** （str，可选）- 该参数供开发人员打印调试信息时使用，具体用法参见 :ref:`api_guide_Name` ，默认值为None。

返回：与 ``x`` 维度相同，数据类型相同的 Variable。

返回类型：Variable

**代码示例：**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.enable_imperative()
    x_data = np.array([True, False, True, False], dtype=np.bool)
    x = paddle.imperative.to_variable(x_data)
    res = paddle.logical_not(x)
    print(res.numpy()) # [False  True False  True]
