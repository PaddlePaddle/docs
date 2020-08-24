.. _cn_api_paddle_framework_manual_seed:

manual_seed
-------------------------------

.. py:function:: paddle.framework.manual_seed(seed)


设置全局默认generator的随机种子。


参数:

     - **seed** (int) - 要设置的的随机种子，推荐使用较大的整数。

返回: 
     Generator：全局默认generator对象。

**代码示例**：

.. code-block:: python

    import paddle
    paddle.manual_seed(102)
