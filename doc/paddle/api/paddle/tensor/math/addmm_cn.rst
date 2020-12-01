.. _cn_api_tensor_addmm:


addmm
-------------------------------

.. py:function:: paddle.addmm(input, x, y, alpha=1.0, beta=1.0, name=None)




计算x和y的乘积，将结果乘以标量alpha，再加上input与beta的乘积，得到输出。其中input与x、y乘积的维度必须是可广播的。

计算过程的公式为：

..  math::
    out = alpha * x * y + beta * input

参数:
    - **input** （Tensor）：输入Tensor input，数据类型支持float32, float64。
    - **x** （Tensor）：输入Tensor x，数据类型支持float32, float64。
    - **y** （Tensor）：输入Tensor y，数据类型支持float32, float64。
    - **alpha** （float，可选）：乘以x*y的标量，数据类型支持float32, float64，默认值为1.0。
    - **beta** （float，可选）：乘以input的标量，数据类型支持float32, float64，默认值为1.0。
    - **name** （str，可选）：具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：计算得到的Tensor。Tensor数据类型与输入input数据类型一致。

返回类型：变量（Tensor）


**代码示例**:

.. code-block:: python

    import paddle

    x = paddle.ones([2,2])
    y = paddle.ones([2,2])
    input = paddle.ones([2,2])

    out = paddle.addmm( input=input, x=x, y=y, beta=0.5, alpha=5.0 )

    print( out.numpy() )
    # [[10.5 10.5]
    # [10.5 10.5]]
