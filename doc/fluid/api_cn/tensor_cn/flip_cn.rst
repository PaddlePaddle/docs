.. _cn_api_tensor_flip:

flip
-------------------------------

.. py:function:: paddle.flip(input, dims, name=None):

:alias_main: paddle.flip
:alias: paddle.flip,paddle.tensor.flip,paddle.tensor.manipulation.flip



该OP沿指定轴反转n维tensor.

参数：
    - **input** (Variable) - 输入Tensor。维度为多维，数据类型为bool, int32, int64, float32或float64。
    - **dims** (list) - 需要翻转的轴。当 ``dims[i] < 0`` 时，实际的计算维度为 rank(input) + dims[i]，其中i为dims的索引。
    - **name** (str|None) - 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。默认值为None。

返回：在指定dims上翻转后的Tensor，与输入input数据类型相同。

返回类型：Variable，与输入input数据类型相同。

抛出异常：
    - ``TypeError`` - 当输出 ``out`` 和输入 ``input`` 数据类型不一致时候。
    - ``ValueError`` - 当参数  ``dims`` 不合法时。

**代码示例1**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    input = paddle.data(name='x', shape=[-1, 2, 2], dtype='float32')
    output = paddle.flip(input, dims=[0, 1])
    exe = paddle.Executor(paddle.CPUPlace())
    exe.run(paddle.default_startup_program())
    img = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = exe.run(paddle.default_main_program(), feed={'x': img}, fetch_list=[
        output])
    print(res)

**代码示例2**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np
    input = paddle.data(name='x', shape=[-1, 2, 2], dtype='float32')
    output = paddle.flip(input, dims=[0, 1])
    exe = paddle.Executor(paddle.CPUPlace())
    exe.run(paddle.default_startup_program())
    img = np.arange(12).reshape((3, 2, 2)).astype(np.float32)
    res = exe.run(paddle.default_main_program(), feed={'x': img}, fetch_list=[
        output])
    print(res)

