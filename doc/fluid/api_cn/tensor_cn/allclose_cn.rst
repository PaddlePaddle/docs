.. _cn_api_tensor_allclose:

allclose
-------------------------------

.. py:function:: paddle.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False, name=None)

:alias_main: paddle.allclose
:alias: paddle.allclose,paddle.tensor.allclose,paddle.tensor.logic.allclose



逐个检查input和other的所有元素是否均满足如下条件：

..  math::
    \left| input - other \right| \leq atol + rtol \times \left| other \right|

该API的行为类似于 :math:`numpy.allclose` ，即当两个待比较Tensor的所有元素均在一定容忍误差范围内视为相等则该API返回True值。

参数:
    - **input** (Variable) - 第一个输入待比较Tensor input。
    - **other** (Variable) - 第二个输入待比较Tensor other。
    - **rtol** (float，可选) - 相对容忍误差，默认值为1e-5。
    - **atol** (float，可选) - 绝对容忍误差，默认值为1e-8。
    - **equal_nan** (bool，可选) - 如果设置为True，则两个NaN数值将被视为相等，默认值为False。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：计算得到的布尔类型单值Tensor。

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    use_cuda = fluid.core.is_compiled_with_cuda()
    a = fluid.data(name="a", shape=[2], dtype='float32')
    b = fluid.data(name="b", shape=[2], dtype='float32')
    result = paddle.allclose(a, b, rtol=1e-05, atol=1e-08,
                            equal_nan=False, name="ignore_nan")
    result_nan = paddle.allclose(a, b, rtol=1e-05, atol=1e-08,
                                equal_nan=True, name="equal_nan")
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x = np.array([10000., 1e-07]).astype("float32")
    y = np.array([10000.1, 1e-08]).astype("float32")
    result_v, result_nan_v = exe.run(
        feed={'a': x, 'b': y},
        fetch_list=[result, result_nan])
    print(result_v, result_nan_v)
    # Output: (array([False]), array([False]))
    x = np.array([10000., 1e-08]).astype("float32")
    y = np.array([10000.1, 1e-09]).astype("float32")
    result_v, result_nan_v = exe.run(
        feed={'a': x, 'b': y},
        fetch_list=[result, result_nan])
    print(result_v, result_nan_v)
    # Output: (array([ True]), array([ True]))
    x = np.array([1.0, float('nan')]).astype("float32")
    y = np.array([1.0, float('nan')]).astype("float32")
    result_v, result_nan_v = exe.run(
        feed={'a': x, 'b': y},
        fetch_list=[result, result_nan])
    print(result_v, result_nan_v)
    # Output: (array([False]), array([ True]))
