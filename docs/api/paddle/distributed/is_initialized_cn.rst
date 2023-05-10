.. _cn_api_distributed_is_initialized:

is_initialized
-------------------------------


.. py:function:: paddle.distributed.is_initialized()

检查分布式环境是否已经被初始化。

参数
:::::::::
无

返回
:::::::::
如果分布式环境初始化完成，默认通信组已完成建立，则返回 True；反之则返回 False。

代码示例
:::::::::
COPY-FROM: paddle.distributed.is_initialized
