.. _cn_api_fluid_layers_addmm:


addmm
-------------------------------

.. py:function:: fluid.layers.addmm(input, x, y, alpha=1.0, beta=1.0, name=None)

计算x和y的乘积，将结果乘以标量alpha，再加上input与beta的乘积，得到输出。其中input与x、y乘积的维度必须是可广播的。

计算过程的公式为：

..  math::
    out = alpha * x * y + beta * input

参数:
    - **input** (Variable) : 输入Tensor input，数据类型支持float32, float64。
    - **x** (Variable) : 输入Tensor x，数据类型支持float32, float64。
    - **y** (Variable) : 输入Tensor y，数据类型支持float32, float64。
    - **alpha** (float，可选) : 乘以x*y的标量，数据类型支持float32, float64，默认值为1.0。
    - **beta** (float，可选) : 乘以input的标量，数据类型支持float32, float64，默认值为1.0。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：计算得到的Tensor。Tensor数据类型与输入input数据类型一致。

返回类型：变量（Variable）


**代码示例**:

.. code-block:: python

    import numpy as np
    import paddle
    import paddle.fluid as fluid

    input = fluid.data(name='input', shape=[2, 2], dtype='float32')
    x = fluid.data(name='x', shape=[2, 2], dtype='float32')
    y = fluid.data(name='y', shape=[2, 2], dtype='float32')
    out = fluid.layers.addmm( input=input, x=x, y=y, alpha=5.0, beta=0.5 )

    data_x = np.ones((2, 2)).astype(np.float32)
    data_y = np.ones((2, 2)).astype(np.float32)
    data_input = np.ones((2, 2)).astype(np.float32)

    place =  fluid.CUDAPlace(0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
    exe = fluid.Executor(place)
    results = exe.run(fluid.default_main_program(), 
                      fetch_list=[out], feed={"input": data_input, 'x': data_x, "y": data_y})
    print(np.array(results[0]))
    # [[10.5 10.5]
    # [10.5 10.5]]
