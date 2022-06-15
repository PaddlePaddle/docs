.. _cn_api_paddle_incubate_autograd_prim2orig:

prim2orig
-------------------------------

.. py:function:: paddle.incubate.autograd.prim2orig(block=None)

.. note::
    只支持在静态图模式下使用。
    参数block必须是None或者是主program的当前block。

对目标程序块中的所有算子进行处理：如果算子是自动微分基础算子，则把该算子替换为一个或者一系列具备等价功能的原生算子，以支持后续执行。


参数
::::::::::::

-- **block** (paddle.static.Block|None, 可选) - 要进行算子替换处理的目标程序块。默认值是 ``None`` ，这时候替换处理发生在主程序的当前程序块上。

返回
::::::::::::
无

代码示例
::::::::::::

.. code-block:: python

    import paddle
    from paddle.incubate.autograd import enable_prim, prim_enabled, prim2orig
    
    paddle.enable_static()
    enable_prim()
    
    x = paddle.ones(shape=[2, 2], dtype='float32')
    x.stop_gradients = False
    y = x * x
    dy_dx = paddle.static.gradients(y, x)
    if prim_enabled():
        prim2orig()
