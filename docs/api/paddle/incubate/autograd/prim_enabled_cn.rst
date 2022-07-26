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

COPY-FROM: paddle.incubate.autograd.prim_enabled