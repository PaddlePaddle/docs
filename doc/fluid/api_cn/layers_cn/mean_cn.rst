.. _cn_api_fluid_layers_mean:

mean
-------------------------------

.. py:function:: paddle.fluid.layers.mean(x, name=None)

计算 ``x`` 所有元素的平均值。

参数：
        - **x** (Variable) : Tensor 或 LoDTensor。均值运算的输入。
        - **name** (basestring | None) : 输出变量的名称。

返回：
        - Variable: 包含输出均值的 Tensor / LoDTensor。

返回类型：
        - Variable（变量）。

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    # Graph Organizing
    input = fluid.layers.data(
        name='data', shape=[2, 3], dtype='float32')
    output = fluid.layers.mean(input)

    # Create an executor using CPU as an example
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # Execute
    x_ndarray = numpy.ones([2, 3]).astype(numpy.float32)
    res, = exe.run(fluid.default_main_program(),
                   feed={'data':x_ndarray},
                   fetch_list=[output])
    print(res)
    '''
    Output Value:
    [1.]
    '''
