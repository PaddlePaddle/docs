.. _cn_api_tensor_equal:

equal
-------------------------------
.. py:function:: paddle.equal(x, y, axis=-1, name=None)

该OP返回 :math:`x==y` 逐元素比较x和y是否相等，所有的元素都相同则返回True，否则返回False。

参数：
    - **x** (Variable) - 输入Tensor，支持的数据类型包括 float32， float64，int32， int64。
    - **y** (Variable) - 输入Tensor，支持的数据类型包括 float32， float64， int32， int64。
    - **axis** (int, 可选) - 如果输入的两个Tensor的维度不相同，并且如果y的维度是x的一部分, 那就可以通过broadcast的方式来进行op计算。axis是进行broadcast的开始的维度，具体broadcast的方式可以参考elementwise_add。 
    - **name** （str，可选）- 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。
    

返回：输出结果的Tensor，输出Tensor只有一个元素值，元素值是True或者False，Tensor数据类型为bool。

返回类型：变量（Variable）

**代码示例**:

.. code-block:: python

    import paddle.fluid as fluid
    import paddle
    import numpy as np
    label = fluid.layers.assign(np.array([3, 4], dtype="int32"))
    label_1 = fluid.layers.assign(np.array([1, 2], dtype="int32"))
    limit = fluid.layers.assign(np.array([3, 4], dtype="int32"))
    out1 = paddle.equal(x=label, y=limit) #out1=[True]
    out2 = paddle.equal(x=label_1, y=limit) #out2=[False]

.. code-block:: python

    import paddle.fluid as fluid
    import paddle
    import numpy as np
    def gen_data():
        return {
              "x": np.ones((2, 3, 4, 5)).astype('float32'),
              "y": np.zeros((3, 4)).astype('float32')
          }
    x = fluid.data(name="x", shape=[2,3,4,5], dtype='float32')
    y = fluid.data(name="y", shape=[3,4], dtype='float32')
    out = paddle.equal(x, y, axis=1)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    res = exe.run(feed=gen_data(),
                      fetch_list=[out])
    print(res[0]) #[False]
