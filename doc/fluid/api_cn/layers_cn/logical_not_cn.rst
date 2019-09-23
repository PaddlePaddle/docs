.. _cn_api_fluid_layers_logical_not:

logical_not
-------------------------------

.. py:function:: paddle.fluid.layers.logical_not(, out=None, name=None)

该OP逐元素的对 ``X`` Tensor进行逻辑非运算

.. math::
        Out = !X

参数：

        - **x** （Variable）- 逻辑非运算的输入，是一个多维的Tensor，数据类型只能是bool。
        - **out** （Variable，可选）- 指定算子输出结果的Tensor，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。
        - **name** （str，可选）- 该参数供开发人员打印调试信息时使用，具体用法参见 :ref:`api_guide_Name` ，默认值为None。

返回：与 ``x`` 维度相同，数据类型相同的Tensor。

返回类型：Variable

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np

    # Graph organizing
    x = fluid.layers.data(name='x', shape=[2], dtype='bool')
    res = fluid.layers.logical_not(x)
    # res = fluid.layers.fill_constant(shape=[2], dtype='bool', value=0)
    # fluid.layers.logical_not(x, out=res)

    # Create an executor using CPU as an example
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # Execute
    x_i = np.array([[1, 0]]).astype(np.bool)
    res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res])
    print(res_val) # [[False, True]]

