.. _cn_api_paddle_set_default_dtype:

set_default_dtype
-------------------------------

.. py:function:: paddle.set_default_dtype(d)


设置默认的全局 dtype。默认的全局 dtype 是 float32。


参数
::::::::::::


     - **d** (string|paddle.dtype|np.dtype) - 设为默认值的 dtype。它仅支持 bfloat16、float16、float32 和 float64。

返回
::::::::::::
 无。

代码示例
::::::::::::

COPY-FROM: paddle.set_default_dtype
