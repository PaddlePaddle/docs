.. _cn_api_paddle_incubate_autograd_prim2orig:

prim2orig
-------------------------------

.. py:function:: paddle.incubate.autograd.prim2orig(block=None)

.. note::
    只支持在静态图模式下使用。
    参数 block 必须是 None 或者是主 program 的当前 block。

对目标程序块中的所有算子进行处理：如果算子是自动微分基础算子，则把该算子替换为一个或者一系列具备等价功能的原生算子，以支持后续执行。


参数
::::::::::::

- **block** (paddle.static.Block|None, 可选) - 要进行算子替换处理的目标程序块。默认值是 ``None`` ，这时候替换处理发生在主程序的当前程序块上。

返回
::::::::::::
无

代码示例
::::::::::::

COPY-FROM: paddle.incubate.autograd.prim2orig
