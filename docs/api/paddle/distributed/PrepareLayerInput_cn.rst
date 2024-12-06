.. _cn_api_paddle_distributed_PrepareLayerInput:

PrepareLayerInput
-------------------------------

.. py:class:: paddle.distributed.PrepareLayerInput(fn=None)

使用用户提供的函数，对标记 Layer 的输入进行处理。


参数
:::::::::
    - **fn** (callable，可选) - 用来处理标记 Layer 输入的函数，该函数需要接受并且仅接受一个参数 process_mesh ，并返回真正用来处理输入的函数。默认为 None。


**代码示例**

COPY-FROM: paddle.distributed.PrepareLayerInput
