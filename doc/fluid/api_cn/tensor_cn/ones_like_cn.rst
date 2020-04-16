.. _cn_api_tensor_ones_like:

ones_like
-------------------------------

.. py:function:: paddle.ones_like(input, dtype=None, device=None, name=None)


该OP创建一个和input具有相同的形状和数据类型的全1Tensor。

参数：
    - **input** (Variable) – 指定输入为一个多维的Tensor，数据类型可以是bool，float32，float64，int32，int64。
    - **dtype** （np.dtype|core.VarDesc.VarType|str， 可选）- 输出变量的数据类型。若参数为空，则输出变量的数据类型和输入变量相同，默认值为None。
    - **device** (str，可选) – 选择在哪个设备运行该操作，可选值包括None，'cpu'和'gpu'。如果 ``device`` 为None，则将选择运行Paddle程序的设备，默认为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    
返回：返回一个存储结果的Tensor。

返回类型：Variable

**代码示例**：

.. code-block:: python

    import paddle
    import paddle.fluid as fluid
    x = fluid.data(name='x', dtype='float32', shape=[3])
    data = paddle.ones_like(x) # data=[1.0, 1.0, 1.0]
    data1 = paddle.ones_like(input=x, device="gpu") # data1=[1.0, 1.0. 1.0]
