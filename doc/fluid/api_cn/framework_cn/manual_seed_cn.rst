.. _cn_api_paddle_framework_manual_seed:

manual_seed
-------------------------------

.. py:function:: paddle.framework.manual_seed(seed)


设置并固定管理随机数生成的默认Generator的随机种子。


参数:

     - **seed** (int) - 设置产生随机数的种子

返回: 
     Generator：默认的generator对象。

**代码示例**：

.. code-block:: python

    import paddle
    paddle.manual_seed(102)
