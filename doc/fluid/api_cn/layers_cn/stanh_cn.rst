.. _cn_api_fluid_layers_stanh:

stanh
-------------------------------

.. py:function:: paddle.fluid.layers.stanh(x, scale_a=0.67, scale_b=1.7159, name=None)

STanh 激活算子（STanh Activation Operator.）

.. math::
          \\out=b*\frac{e^{a*x}-e^{-a*x}}{e^{a*x}+e^{-a*x}}\\

参数：
    - **x** (Tensor|LoDTensor) - 数据类型为float32,float64。STanh operator的输入
    - **scale_a** (float) - 输入的a的缩放参数
    - **scale_b** (float) - b的缩放参数
    - **name** (str|None) - 这个层的名称(可选)。如果设置为None，该层将被自动命名

返回: 与输入shape相同的张量

返回类型: Variable（Tensor），数据类型为float32的Tensor。

**代码示例：**

.. code-block:: python

    import paddle.fluid as fluid
    import numpy as np
    data = fluid.layers.data(name="input", shape=[-1, 3])
    result = fluid.layers.stanh(data,scale_a=0.67, scale_b=1.72)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    x = np.random.random(size=(3, 3)).astype('float32')
    output= exe.run(feed={"input": x},
                 fetch_list=[result])
    print(output)
    """
    output:
    [array([[0.626466  , 0.89842904, 0.7501062 ],
           [0.25147712, 0.7484996 , 0.22902708],
           [0.62705994, 0.23110689, 0.56902856]], dtype=float32)]
    """


