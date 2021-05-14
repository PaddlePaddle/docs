.. _cn_api_tensor_increment:

increment
-------------------------------

.. py:function:: paddle.increment(x, value=1.0, name=None)




该OP在控制流程中用来让 ``x`` 的数值增加 ``value`` 。

**参数**:
  - **x** (Tensor) – 输入张量，必须始终只有一个元素。支持的数据类型：float32，float64，int32，int64。
  - **value** (float，可选) – ``x`` 的数值增量。默认值为1.0。
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为None。

**返回**: Tensor，形状和数据类型同输入 ``x`` 。


**代码示例**:

.. code-block:: python

    import paddle

    data = paddle.zeros(shape=[1], dtype='float32')
    counter = paddle.increment(data)
    # [1.]
