.. _cn_api_paddle_tensor_log1p:

log1p
-------------------------------

.. py:function:: paddle.tensor.log1p(x, out=None, name=None)


该OP计算Log1p（加一的自然对数）结果。

.. math::
                  \\Out=ln(x+1)\\


参数:
  - **x** (Variable) – 该OP的输入为LodTensor/Tensor。数据类型为float32，float64。 
  - **out**  (Variable， 可选) -  指定算子输出结果的LoDTensor/Tensor，可以是程序中已经创建的任何Variable。默认值为None，此时将创建新的Variable来保存输出结果。
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

返回：Log1p算子自然对数输出

返回类型: Variable - 该OP的输出为LodTensor/Tensor，数据类型为输入一致。


**代码示例**

..  code-block:: python

 import paddle
 import paddle.fluid as fluid
 import numpy as np

 x = fluid.data(name="x", shape=[2,1], dtype="float32")
 res = paddle.log1p(x) # paddle.log1p等价于 paddle.tensor.log1p

 # 举例选择CPU计算环境
 exe = fluid.Executor(fluid.CPUPlace())

 # 执行静态图，输出结果
 x_i = np.array([[0], [1]]).astype(np.float32)
 res_val, = exe.run(fluid.default_main_program(), feed={'x':x_i}, fetch_list=[res])
 print(res_val) # [[0.], [0.6931472]]


