.. _cn_api_paddle_Tensor_create_tensor:

paddle.Tensor.create_tensor
-------------------------------

.. py:function:: paddle.Tensor.create_tensor(dtype, name=None, persistable=False)

根据数据类型 dtype 创建一个 Tensor。

参数
::::::::::::
- dtype (string|numpy.dtype) – 要创建的 Tensor 数据类型，可以是如下值： bool, float16, float32, float64, int8, int16, int32 和 int64。
- name (string, 可选) – 默认值为 None。通常用户不需要设置此属性。有关详细信息，请参阅 :ref:`api_guide_Name`。
- persistable (bool) – 设置创建的张量是否持久。默认值为 False。

返回
::::::::::::
Tensor，数据类型为指定的 dtype。

代码示例
::::::::::::

COPY-FROM: paddle.Tensor.create_tensor
