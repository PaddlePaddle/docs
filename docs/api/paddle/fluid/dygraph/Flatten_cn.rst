.. _cn_api_tensor_Flatten:

Flatten
-------------------------------

.. py:function:: paddle.nn.Flatten(start_axis=1, stop_axis=-1)



该接口用于构造一个 ``Flatten`` 类的可调用对象。更多信息请参见代码示例。它实现将一个连续维度的Tensor展平成一维Tensor。


参数：
    - start_axis (int，可选) - 展开的起始维度，默认值为1。
    - stop_axis  (int，可选) - 展开的结束维度，默认值为-1。

返回：  无。


**代码示例**

..  code-block:: python

    import paddle
    import numpy as np

    inp_np = np.ones([5, 2, 3, 4]).astype('float32')
    inp_np = paddle.to_tensor(inp_np)
    flatten = paddle.nn.Flatten(start_axis=1, stop_axis=2)
    flatten_res = flatten(inp_np)
