.. _cn_api_paddle_incubate_autograd_enable_prim:

enable_prim
-------------------------------

.. py:function:: paddle.incubate.autograd.enable_prim()

.. note::
    只支持在静态图模式下使用。

开启基于自动微分基础算子的自动微分机制。


返回
::::::::::::
无

代码示例
::::::::::::

.. code-block:: python

    import paddle
    from paddle.incubate.autograd import enable_prim, prim_enabled
    
    paddle.enable_static()
    enable_prim()

    print(prim_enabled()) # True
