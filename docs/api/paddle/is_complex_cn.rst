.. _cn_api_paddle_is_complex:

is_complex
-------------------------------

.. py:function:: paddle.is_complex(x)


判断输入 tensor 的数据类型是否为复数类型(complex64 或者 complex128)。

参数
:::::::::
   - **x** (Tensor) - 输入 Tensor

抛出异常
:::::::::
  - TypeError: 如果输入 ``x`` 不是一个 Tensor.
    

返回
:::::::::
bool, 判断输入 tensor 的数据类型是否为复数类型(complex64 或者 complex128)


代码示例
:::::::::

COPY-FROM: paddle.is_complex