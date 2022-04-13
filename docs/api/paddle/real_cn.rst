.. _cn_api_tensor_real:

real
------

.. py:function:: paddle.real(x, name=None)

返回一个包含输入复数Tensor的实部数值的新Tensor。也可以对一个复数Tensor对象直接调用real方法返回该复数Tensor的实部数值，具体见代码示例。

参数
::::::::::::

    - **x** (Tensor) - 输入Tensor，其数据类型可以为complex64或complex128。
    - **name** (str，可选) - 具体用法请参见 :ref:`api_guide_Name` ，一般无需设置，默认值为None。

返回
::::::::::::
Tensor，包含原复数Tensor的实部数值。

代码示例
::::::::::::
COPY-FROM: <real>:<code-example1>