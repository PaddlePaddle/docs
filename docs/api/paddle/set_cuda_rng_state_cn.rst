.. _cn_api_paddle_set_cuda_rng_state:

set_cuda_rng_state
-------------------------------

.. py:function:: paddle.set_cuda_rng_state(state_list)


设置 cuda 随机数生成器的状态信息。


参数
::::::::::::


     - **state_list** (list [GeneratorState]) - 需要设置的随机数生成器状态信息列表，通过 get_cuda_rng_state() 获取。

返回
::::::::::::

     无。

代码示例
::::::::::::

COPY-FROM: paddle.set_cuda_rng_state
