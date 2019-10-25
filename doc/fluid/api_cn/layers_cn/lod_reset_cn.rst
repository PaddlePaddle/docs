.. _cn_api_fluid_layers_lod_reset:

lod_reset
-------------------------------

.. py:function:: paddle.fluid.layers.lod_reset(x, y=None, target_lod=None)

根据给定的参数 ``y`` 或 ``target_lod`` ，重设输入 ``x`` (LoDTensor) 的 LoD 信息。

参数：
    - **x** (Variable) : 输入变量，类型为 Tensor 或者 LoDTensor。
    - **y** (Variable|None) : 当 ``y`` 非空时，输出 LoDTensor 的 LoD 信息将与 ``y`` 的 LoD 一致。
    - **target_lod** (list|tuple|None) : 一级 LoD，当 ``y`` 为空时，输出 LoDTensor 的 LoD 信息将与 ``target_lod`` 一致。

返回:
    - Variable (LoDTensor)，重设了 LoD 信息的 LoDTensor。

返回类型：
    - Variable (LoDTensor)。

抛出异常：
    - ``TypeError`` : 当 ``y`` 和 ``target_lod`` 二者均为空时抛出此异常。

::

    * 例 1:

    x: 包含一级 LoD 信息的 LoDTensor
        x.lod =  [[ 2,           3,                   1 ]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    y: None

    target_lod: [4, 2]

    Output: 包含一级 LoD 信息的 LoDTensor
        out.lod =  [[4,                          2]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

    * 例 2:

    x: 包含一级 LoD 信息的 LoDTensor
        x.lod =  [[2,            3,                   1]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    y: 普通 Tensor，不含 LoD 信息
        y.data = [[2, 4]]
        y.dims = [1, 3]

    target_lod: 当 y 不为空时，此参数不起作用

    Output: 包含一级 LoD 信息的 LoDTensor
        out.lod =  [[2,            4]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

    * 例 3:

    x: 包含一级 LoD 信息的 LoDTensor
        x.lod =  [[2,            3,                   1]]
        x.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        x.dims = [6, 1]

    y: 包含二级 LoD 信息的 LoDTensor
        y.lod =  [[2, 2], [2, 2, 1, 1]]
        y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
        y.dims = [6, 1]

    target_lod: 当 y 不为空时，此参数不起作用

    Output: 包含二级 LoD 信息的 LoDTensor
        out.lod =  [[2, 2], [2, 2, 1, 1]]
        out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
        out.dims = [6, 1]

**代码示例**：

.. code-block:: python

    import paddle.fluid as fluid
    import numpy

    # Graph Organizing
    x = fluid.layers.data(name='x', shape=[6])
    y = fluid.layers.data(name='y', shape=[6], lod_level=2)
    output = fluid.layers.lod_reset(x=x, y=y)

    # Create an executor using CPU as an example
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # Execute
    x_tensor = fluid.core.LoDTensor()
    x_tensor.set(numpy.ones([6]).astype(numpy.float32), place)
    y_ndarray = numpy.ones([6]).astype(numpy.float32)
    y_lod = [[2, 2], [2, 2, 1, 1]]
    y_tensor = fluid.create_lod_tensor(y_ndarray, y_lod, place)

    res, = exe.run(fluid.default_main_program(),
                   feed={'x':x_tensor, 'y':y_tensor},
                   fetch_list=[output],
                   return_numpy=False)
    print(res)
    # Output Value:
    # lod: [[0, 2, 4], [0, 2, 4, 5, 6]]
    # dim: 6
    # layout: NCHW
    # dtype: float
    # data: [1 1 1 1 1 1]
