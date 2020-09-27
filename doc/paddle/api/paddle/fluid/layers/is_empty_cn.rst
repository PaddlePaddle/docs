.. _cn_api_fluid_layers_is_empty:

is_empty
-------------------------------

.. py:function:: paddle.is_empty(x, name=None)




测试变量是否为空

参数：
    - **x** (Tensor)-测试的变量
   - **name** （str，可选）- 输出的名字。默认值为None。该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` 。

返回：布尔类型的标量。如果变量x为空则值为真

返回类型：Tensor


**代码示例**：

.. code-block:: python

    # dygraph_mode
    import paddle

    input = paddle.rand(shape=[4, 32, 32], dtype='float32')
    res = paddle.is_empty(x=input)
    print("res:", res)
    # ('res:', Tensor: eager_tmp_1
    #    - place: CPUPlace
    #    - shape: [1]
    #    - layout: NCHW
    #    - dtype: bool
    #    - data: [0])


.. code-block:: python

    # static mode
    import numpy as np
    import paddle

    paddle.enable_static()
    input = paddle.static.data(name="input", shape=[4, 32, 32], dtype="float32")
    res = paddle.is_empty(x=input)

    exe = paddle.static.Executor(paddle.CPUPlace())
    data = np.ones((4, 32, 32)).astype(np.float32)
    out = exe.run(feed={'input':data}, fetch_list=[res])
    print("is_empty: ", out)
    # ('out:', [array([False])])



