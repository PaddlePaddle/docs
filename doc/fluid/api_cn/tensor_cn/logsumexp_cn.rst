.. _cn_api_paddle_tensor_math_logsumexp:

logsumexp
-------------------------------

.. py:function:: paddle.tensor.math.logsumexp(x, dim=None, keepdim=False, out=None, name=None)

该OP对输入Tensor的元素以e为底做指数运算，然后根据指定维度做求和之后取自然对数

.. math::
   logsumexp(x) = \log\sum exp(x)

参数：
          - **x** （Variable）- 输入变量为多维Tensor或LoDTensor，支持数据类型为float32，float64
          - **dim** （list | int ，可选）- 求和运算的维度。如果为None，则计算所有元素的和并返回包含单个元素的Tensor变量，否则必须在  :math:`[−rank(input),rank(input)]` 范围内。如果 :math:`dim [i] <0` ，则维度将变为 :math:`rank+dim[i]` ，默认值为None。
          - **keep_dim** （bool）- 是否在输出Tensor中保留减小的维度。如 keep_dim 为true，否则结果张量的维度将比输入张量小，默认值为False。
          - **out** （Variable ， 可选）- 显示指定的输出变量
          - **name** （str ， 可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：  Tensor，数据类型和输入数据类型一致。

返回类型：Variable

**代码示例1**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard():
      np_x = np.random.uniform(0.1, 1, [10]).astype(np.float32)
      x = fluid.dygraph.to_variable(np_x)
      print(paddle.logsumexp(x).numpy())

**代码示例2**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    with fluid.dygraph.guard():
        np_x = np.random.uniform(0.1, 1, [2, 3, 4]).astype(np.float32)
        x = fluid.dygraph.to_variable(np_x)
        print(paddle.logsumexp(x, dim=1).numpy())
        print(paddle.logsumexp(x, dim=[0, 2]).numpy())
