.. _cn_api_tensor_clamp:

clamp
-------------------------------

.. py:function:: paddle.clamp(input, min=None, max=None, output=None, name=None)

该OP将输入的所有元素进行剪裁，使得输出元素限制在[min, max]内，具体公式如下：

.. math::

        Out = MIN(MAX(x, min), max) 

参数：
    - **input** (Variable) – 指定输入为一个多维的Tensor，数据类型可以是float32，float64。
    - **min** (float32|Variable, 可选) - 裁剪的最小值，输入中小于该值的元素将由该元素代替，若参数为空，则不对输入的最小值做限制。数据类型可以是float32或形状为[1]的Tensor，类型可以为int32，float32，float64，默认值为None。
    - **max** (float32|Variable, 可选) - 裁剪的最大值，输入中大于该值的元素将由该元素代替，若参数为空，则不对输入的最大值做限制。数据类型可以是float32或形状为[1]的Tensor，类型可以为int32，float32，float64，默认值为None。
    - **output** （Variable， 可选）- 输出Tensor或LoDTensor。如果为None，则创建一个新的Tensor作为输出Tensor，默认值为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    
返回：返回一个和输入形状相同的Tensor。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    in1 = np.array([[1.2,3.5],
                    [4.5,6.4]]).astype('float32')
    with fluid.dygraph.guard():
        x1 = fluid.dygraph.to_variable(in1)
        out1 = paddle.tensor.clamp(x1, min=3.5, max=5.0)
        out2 = paddle.tensor.clamp(x1, min=2.5)
        print(out1.numpy())
        # [[3.5, 3.5]
        # [4.5, 5.0]]
        print(out2.numpy())
        # [[2.5, 3.5]
        # [[4.5, 6.4]

