.. _cn_api_paddle_framework_manual_seed:

manual_seed
-------------------------------

.. py:function:: paddle.framework.manual_seed(seed)

:alias_main: paddle.manual_seed
:alias: paddle.manual_seed,paddle.framework.random.manual_seed




设置并固定随机种子, manual_seed设置后，会将用户定义的Program中的random_seed参数设置成相同的种子


参数:

     - **seed** (int32|int64) - 设置产生随机数的种子

返回: 无

**代码示例**：

.. code-block:: python

    import paddle
    from paddle.framework import manual_seed
    
    default_seed = paddle.default_startup_program().random_seed
    
    manual_seed(102)
    prog = paddle.Program()
    prog_seed = prog.random_seed
    update_seed = paddle.default_startup_program().random_seed

