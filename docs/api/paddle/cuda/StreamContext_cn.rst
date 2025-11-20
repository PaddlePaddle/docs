.. _cn_api_paddle_cuda_StreamContext:

StreamContext
-------------

.. py:class:: paddle.cuda.StreamContext(stream)

该上下文管理器用于临时切换当前 CUDA 流，离开上下文后自动恢复之前的流。

参数：
::::::::::::
    - **stream** (paddle.cuda.Stream) - 要切换到的 CUDA 流对象

代码示例
::::::::::::
COPY-FROM: paddle.cuda.StreamContext
