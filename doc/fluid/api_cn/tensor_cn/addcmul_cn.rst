.. _cn_api_tensor_addcmul:

addcmul
-------------------------------

.. py:function:: paddle.addcmul(input, tensor1, tensor2, value=1.0, out=None, name=None)

计算tensor1和tensor2的逐元素乘积，然后将结果乘以标量value，再加到input上输出。其中input, tensor1, tensor2的维度必须是可广播的。

计算过程的公式为：
..  math::
    out = input + value * tensor1 * tensor2

参数:
    - **input** (Variable) : 输入Tensor input，数据类型支持float32, float64, int32, int64。
    - **itensor1** (Variable) : 输入Tensor tensor1，数据类型支持float32, float64, int32, int64。
    - **itensor2** (Variable) : 输入Tensor tensor2，数据类型支持float32, float64, int32, int64。
    - **value** (int|float) : 乘以tensor1*tensor2的标量。如果输入input类型为float32或float64，value类型必须为float，如果输入input类型为int32或int64，value类型必须为int。
    - **out** (Variable, 可选) – 指定存储运算结果的Tensor。如果设置为None或者不设置，将创建新的Tensor存储运算结果，默认值为None。
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回：计算得到的Tensor。Tensor数据类型与输入input数据类型一致。

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    import paddle
    import paddle.fluid as fluid

    input = fluid.data(name='input', dtype='float32', shape=[3, 4])
    tensor1 = fluid.data(name='tenosr1', dtype='float32', shape=[1, 4])
    tensor2 = fluid.data(name='tensor2', dtype='float32', shape=[3, 4])
    data = paddle.addcmul(input, tensor1, tensor2, value=1.0)

