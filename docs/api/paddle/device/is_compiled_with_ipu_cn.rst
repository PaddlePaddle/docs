.. _cn_api_fluid_is_compiled_with_ipu:

is_compiled_with_ipu
-------------------------------

.. py:function:: paddle.device.is_compiled_with_ipu()




检查 ``whl`` 包是否可以被用来在 Graphcore IPU 上运行模型

返回
::::::::::
    bool，支持 Graphcore IPU 则为 True，否则为 False。

代码示例
::::::::::

COPY-FROM: paddle.device.is_compiled_with_ipu
