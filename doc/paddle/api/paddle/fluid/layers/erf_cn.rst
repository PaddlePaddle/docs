.. _cn_api_fluid_layers_erf:

erf
-------------------------------

.. py:function:: paddle.erf(x,name = None)




逐元素计算 Erf 激活函数。更多细节请参考 `Error function <https://en.wikipedia.org/wiki/Error_function>`_ 。


.. math::
    out = \frac{2}{\sqrt{\pi}} \int_{0}^{x}e^{- \eta^{2}}d\eta

参数：
  - **x** (Tensor) - Erf Op 的输入，多维 Tensor 或 LoDTensor，数据类型为 float16, float32 或 float64。
  - **name** (str，可选) – 该参数供开发人员打印调试信息时使用，具体用法请参见 :ref:`api_guide_Name` ，默认值为 None。

返回：
  - 多维 Tensor 或 LoDTensor, 数据类型为 float16, float32 或 float64， 和输入 x 的数据类型相同，形状和输入 x 相同。


**代码示例**：

.. code-block:: python

    import numpy as np
    import paddle

    x = paddle.rand([1,3],dtype = "float32")
    y = paddle.erf(x)
