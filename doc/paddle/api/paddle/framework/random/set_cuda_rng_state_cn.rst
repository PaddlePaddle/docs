.. _cn_api_paddle_cn_set_cuda_rng_state:

set_cuda_rng_state
-------------------------------

.. py:function:: paddle.set_cuda_rng_state(state_list)


设置cuda随机数生成器的状态信息


参数:

     - **state_list** (list [GeneratorState]) - 需要设置的随机数生成器状态信息列表，通过get_cuda_rng_state()获取。

返回: 
     无

**代码示例**：

.. code-block:: python

    import paddle
    sts = paddle.get_cuda_rng_state()
    paddle.set_cuda_rng_state(sts)
