.. _cn_api_paddle_is_compiled_with_rocm:

is_compiled_with_rocm
-------------------------------

.. py:function:: paddle.device.is_compiled_with_rocm()




检查 ``whl`` 包是否可以被用来在AMD或海光GPU(ROCm)上运行模型。

返回
::::::::::::
bool，支持GPU(ROCm)则为True，否则为False。

代码示例
::::::::::::

COPY-FROM: paddle.device.is_compiled_with_rocm
