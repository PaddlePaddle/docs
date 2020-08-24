.. _cn_api_fluid_layers_logical_or:

logical_or
-------------------------------

.. py:function:: paddle.logical_or(x, y, out=None, name=None)

:alias_main: paddle.logical_or
:alias: paddle.logical_or, paddle.tensor.logical_or, paddle.tensor.logic.logical_or
:old_api: paddle.fluid.layers.logical_or



该OP逐元素的对 ``X`` 和 ``Y`` 进行逻辑或运算。

.. math::
        Out = X || Y

.. note::
    ``paddle.logical_or`` 遵守broadcasting，如您想了解更多，请参见 :ref:`cn_user_guide_broadcasting` 。

参数：
        - **x** （Variable）- 逻辑或运算的第一个输入，是一个 Variable，数据类型只能是bool。
        - **y** （Variable）- 逻辑或运算的第二个输入，是一个 Variable，数据类型只能是bool。
        - **out** （Variable，可选）- 指定算子输出结果的 Variable，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。
        - **name** （str，可选）- 该参数供开发人员打印调试信息时使用，具体用法参见 :ref:`api_guide_Name` ，默认值为None。

返回：与 ``x`` 维度相同，数据类型相同的 Variable。

返回类型：Variable


**代码示例：**

.. code-block:: python

    import paddle
    import numpy as np

    paddle.enable_imperative()
    x_data = np.array([True, True, False, False], dtype=np.bool)
    y_data = np.array([True, False, True, False], dtype=np.bool)
    x = paddle.imperative.to_variable(x_data)
    y = paddle.imperative.to_variable(y_data)
    res = paddle.logical_or(x, y)
    print(res.numpy()) # [True  True  True False]
