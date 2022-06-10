.. _cn_api_paddle_incubate_autograd_prim_enabled:

prim_enabled
-------------------------------

.. py:function:: paddle.incubate.autograd.prim_enabled()

.. note::
    只支持在静态图模式下使用。

显示是否开启了基于自动微分基础算子的自动微分机制。默认状态是关闭。


返回
::::::::::::

- **flag** (bool) - 是否开启了基于自动微分基础算子的自动微分机制。

代码示例
::::::::::::

.. code-block:: python

    import paddle
    from paddle.incubate.autograd import enable_prim, disable_prim, prim_enabled
    
    paddle.enable_static()
    enable_prim()

    print(prim_enabled()) # True

    disable_prim()

    print(prim_enabled()) # False
