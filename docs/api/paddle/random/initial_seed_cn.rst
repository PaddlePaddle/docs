.. _cn_api_paddle_random_initial_seed:

initial_seed
-------------------------------

.. py:function:: paddle.random.initial_seed()

获取当前随机数生成器的初始种子值。

该函数返回用于初始化随机数生成器的种子值。这个种子值决定了随机数生成器的初始状态，相同的种子会产生相同的随机数序列。

返回
:::::::::
int：当前随机数生成器的初始种子值。

代码示例
::::::::::

COPY-FROM: paddle.random.initial_seed
