.. _cn_api_tensor_random_randn:

randn
-------------------------------

.. py:function:: paddle.tensor.random.randn(shape, out=None, dtype=None, device=None, stop_gradient=True, name=None)

该 API 用于生成数据符合标准正态随机分布（均值为 0，方差为 1 的正态随机分布）的 Tensor。

参数：
  - **shape** (list|tuple): 生成的随机 Tensor 的形状。
  - **out** (Variable, optional): 用于存储创建的 Tensor，可以是程序中已经创建的任何Variable。当该参数值为 `None` 时，将创建新的 Variable 来保存输出结果。默认值为 None。
  - **dtype** (np.dtype|core.VarDesc.VarType|str, optional): 输出 Tensor 的数据类型，可选值为 float32，float64。当该参数值为 `None` 时， 输出当 Tensor 的数据类型为 `float32` 。默认值为 None.
  - **device** (str, optional): 用于指定输出变量是保存在 CPU 还是 GPU 内存中。可选值为 None，'cpu'，'gpu'。当该参数为 None 时， 输出变量将会自动的分配到相对应内存中。默认值为 None。
  - **stop_gradient** (bool, optional): 是否停止输出当前变量（输出变量）的梯度值。默认值为 True。
  - **name** (str, optional): 该参数供开发人员打印调试信息时使用，具体用法参见 :ref:`api_guide_Name` ，默认值为None。

返回：符合标准正态分布的随机 Tensor。形状为 shape，数据类型为 dtype。

返回类型：Variable

**示例代码**

.. code-block:: python

     # declarative mode
     import paddle
     import paddle.fluid as fluid
     data = paddle.randn([2, 4])
     place = fluid.CPUPlace()
     exe = fluid.Executor(place)
     res, = exe.run(fluid.default_main_program(), feed={}, fetch_list=[data])
     print(res)
     # [[-1.4187592   0.7368311  -0.53748125 -0.0146909 ]
     #  [-0.66294265 -1.3090698   0.1898754  -0.14065823]]

.. code-block:: python

    # imperative mode
    import paddle
    import paddle.fluid as fluid
    import paddle.fluid.dygraph as dg
    place = fluid.CPUPlace()
    with dg.guard(place) as g:
        x = paddle.randn([2, 4])
        x_np = x.numpy()
        print(x_np)
        # [[ 1.5149173  -0.26234224 -0.592486    1.4523455 ]
        #  [ 0.04581212 -0.85345626  1.1687907  -0.02512913]]
