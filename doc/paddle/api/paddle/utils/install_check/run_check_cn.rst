.. _cn_api_paddle_utils_run_check:

run_check
-------------------------------

.. py:function:: paddle.utils.run_check()

检查用户机器上，PaddlePaddle是否正确地安装了，以及是否能够成功运行。


代码示例
::::::::::

.. code-block:: python

    import paddle

    paddle.utils.run_check()
    # Running verify PaddlePaddle program ...
    # W1010 07:21:14.972093  8321 device_context.cc:338] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 11.0, Runtime API Version: 10.1
    # W1010 07:21:14.979770  8321 device_context.cc:346] device: 0, cuDNN Version: 7.6.
    # PaddlePaddle works well on 1 GPU.
    # PaddlePaddle works well on 8 GPUs.
    # PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

