.. _cn_api_tensor_histogram:

histogram
-------------------------------

.. py:function:: paddle.histogram(input, bins=100, min=0, max=0):

计算输入张量的直方图。以min和max为range边界，将其均分成bins个直条，然后将排序好的数据划分到各个直条(bins)中。如果min和max都为0, 则利用数据中的最大最小值作为边界。

参数：
    - **input** (Variable) - 输入Tensor。维度为多维，数据类型为int32, int64, float32或float64。
    - **bins** (int) - 直方图 bins(直条)的个数，默认为100。
    - **min** (int) - range的下边界(包含)，默认为0。
    - **max** (int) - range的上边界(包含)，默认为0。

返回：直方图。

返回类型：Variable，数据为int64类型，维度为(nbins,)。

抛出异常：
    - ``ValueError`` - 当输入 ``bin``, ``min``, ``max``不合法时。

**代码示例1**：

.. code-block:: python

    import paddle
    import numpy as np
    startup_program = paddle.Program()
    train_program = paddle.Program()
    with paddle.program_guard(train_program, startup_program):
        inputs = paddle.data(name='input', dtype='int32', shape=[2,3])
        output = paddle.histogram(inputs, bins=5, min=1, max=5)
        place = paddle.CPUPlace()
        exe = paddle.Executor(place)
        exe.run(startup_program)
        img = np.array([[2, 4, 2], [2, 5, 4]]).astype(np.int32)
        res = exe.run(train_program,
                      feed={'input': img},
                      fetch_list=[output])
        print(np.array(res[0])) # [0, 3, 0, 2, 1]

**代码示例2**：

.. code-block:: python

    import paddle
    import numpy as np
    with paddle.imperative.guard(paddle.CPUPlace()):
        inputs_np = np.array([0.5, 1.5, 2.5]).astype(np.float)
        inputs = paddle.imperative.to_variable(inputs_np)
        result = paddle.histogram(inputs, bins=5, min=1, max=5)
        print(result) # [1, 1, 0, 0, 0]
