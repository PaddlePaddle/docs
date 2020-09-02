.. _cn_api_fluid_layers_triu:

triu
-------------------------------


.. py:function:: paddle.fluid.layers.triu(input, diagonal=0, name=None)
:api_attr: 声明式编程模式（静态图)



上三角矩阵（Triangular Upper）

上三角矩阵函数返回矩阵(2-D tensor) 或批矩阵 `input` 的上三角矩阵, 返回矩阵中的其他元素被置为0. 矩阵的上三角矩阵的是指: 包含原矩阵对角线上方(包括对角线上)元素的矩阵.

更多详情请参考 : `Triangular matrix <https://en.wikipedia.org/wiki/Triangular_matrix>`_

参数：
    - **input** (Variable) - 输入矩阵, 数据类型 float64， float32， int32， int64的Tensor。
    - **diagonal** (int, optional)- 参考的矩阵对角线，默认值是 0。 如果diagonal = 0，参考对角线为主对角线，将保留参考对角线上和对角线往上的所有元素；当 diagonal 为一个大于 0的值 n时，表示实际参考的对角线为矩阵主对角线往上的第 n 条对角线；当 diagonal 为一个小于 0的值 n时，表示实际参考的对角线为矩阵主对角线往下的第 n 条对角线。 主对角线由矩阵中下标为 {(i,i)}， i∈[0,min{d1,d2}−1] 的元素组成，其中 d1,d2 为矩阵的维度。    
    - **name** (str, optional) - 默认值是 None，用户通常不需要设置。 更多信息请参见 :ref:`api_guide_Name`。

返回： 对输入矩阵按指定的参考对角线取得的上三角矩阵，数据类型和输入矩阵相同。 

返回类型：Variable

抛出异常：
    - :code:`TypeError`：当 ``diagonal`` 的数据类型不是int时。
    - :code:`TypeError`：当  输入 Tensor 的维度 ``dimension`` 小于2 时。

**代码示例**：

.. code-block:: python
    import numpy as np
    import paddle.fluid as fluid
    data = np.arange(1, 13, dtype="int64").reshape(3,-1)
    # array([[ 1,  2,  3,  4],
    #        [ 5,  6,  7,  8],
    #        [ 9, 10, 11, 12]])
    x = fluid.data(shape=(-1, 4), dtype='int64', name='x')
    exe = fluid.Executor(fluid.CPUPlace())
    # example 1, default diagonal
    import paddle.fluid as fluid
    triu = fluid.layers.triu(x)
    triu_out, = exe.run(fluid.default_main_program(), feed={"x": data},
        fetch_list=[triu], return_numpy=True)
    # array([[ 1,  2,  3,  4],
    #        [ 0,  6,  7,  8],
    #        [ 0,  0, 11, 12]])
.. code-block:: python
    # example 2, positive diagonal value
    import paddle.fluid as fluid
    import numpy as np
    data = np.arange(1, 13, dtype="int64").reshape(3,-1)
    x = fluid.data(shape=(-1, 4), dtype='int64', name='x')
    exe = fluid.Executor(fluid.CPUPlace())
    triu = fluid.layers.triu(x, diagonal=2)
    triu_out, = exe.run(fluid.default_main_program(), feed={"x": data},
        fetch_list=[triu], return_numpy=True)
    # array([[0, 0, 3, 4],
    #        [0, 0, 0, 8],
    #        [0, 0, 0, 0]])
.. code-block:: python
    # example 3, negative diagonal value
    import paddle.fluid as fluid
    import numpy as np
    data = np.arange(1, 13, dtype="int64").reshape(3,-1)
    x = fluid.data(shape=(-1, 4), dtype='int64', name='x')
    exe = fluid.Executor(fluid.CPUPlace())
    triu = fluid.layers.triu(x, diagonal=-1)
    triu_out, = exe.run(fluid.default_main_program(), feed={"x": data},
        fetch_list=[triu], return_numpy=True)
    # array([[ 1,  2,  3,  4],
    #        [ 5,  6,  7,  8],
    #        [ 0, 10, 11, 12]])
