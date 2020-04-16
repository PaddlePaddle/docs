.. _cn_api_tensor_argmax:

argmax
-------------------------------

.. py:function:: paddle.argmax(input, axis=None, dtype=None, out=None, keepdims=False, name=None)


该OP沿 ``axis`` 计算输入 ``input`` 的最大元素的索引。

参数：
    - **input** (Variable) - 输入的多维 ``Tensor`` ，支持的数据类型：float32、float64、int8、int16、int32、int64。
    - **axis** (int，可选) - 指定对输入Tensor进行运算的轴， ``axis`` 的有效范围是[-R, R)，R是输入 ``input`` 的Rank， ``axis`` -R与绝对值相同的R等价。默认值为0。
    - **dtype** (np.dtype|core.VarDesc.VarType|str)- 输出Tensor的数据类型，可选值为int32，int64，默认值为None，将返回int64类型的结果。
    - **out** (Variable, 可选) – 指定存储运算结果的Tensor。如果设置为None或者不设置，将创建新的Tensor存储运算结果，默认值为None。
    - **keepdims** （bool，可选）- 是否保留进行max index操作的维度，默认值为False。
    - **name** (str，可选) – 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回： ``Tensor`` ，数据类型int64

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    import numpy as np

    in1 = np.array([[[5,8,9,5],
                     [0,0,1,7],
                     [6,9,2,4]],
                    [[5,2,4,2],
                     [4,7,7,9],
                     [1,7,0,6]]])
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(in1)
        out1 = paddle.argmax(input=x, axis=-1)
        out2 = paddle.argmax(input=x, axis=0)
        out3 = paddle.argmax(input=x, axis=1)
        out4 = paddle.argmax(input=x, axis=2)
        out5 = paddle.argmax(input=x, axis=2, keepdims=True)
        print(out1.numpy())
        # [[2 3 1]
        #  [0 3 1]]
        print(out2.numpy())
        # [[0 0 0 0]
        #  [1 1 1 1]
        #  [0 0 0 1]]
        print(out3.numpy())
        # [[2 2 0 1]
        #  [0 1 1 1]]
        print(out4.numpy())
        # [[2 3 1]
        #  [0 3 1]]
        print(out5.numpy())
        #array([[[2],
        #        [3],
        #        [1]],
        #       [[0],
        #        [3],
        #        [1]]])
