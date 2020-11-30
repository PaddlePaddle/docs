.. _cn_api_paddle_framework_get_cuda_rng_state:

get_cuda_rng_state
-------------------------------

.. py:function:: paddle.framework.get_cuda_rng_state()


获取cuda随机数生成器的状态信息


参数:

     无

返回: 
     GeneratorState：对象

**代码示例**：

.. code-block:: python

    import paddle
    sts = paddle.get_cuda_rng_state()
