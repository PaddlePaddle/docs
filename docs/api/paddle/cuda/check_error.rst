.. _cn_api_paddle_cuda_check_error:

check_error
-----------

.. py:function:: paddle.cuda.check_error(res)

检查给定的 CUDA 运行时返回的状态码，当 ``res`` 是错误码时，抛出对应的 CUDA 错误异常。

参数
::::::::::::

    - **res** (int) - CUDA 运行时返回的状态码。

代码示例
::::::::::::
COPY-FROM: paddle.cuda.check_error
