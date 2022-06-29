.. _cn_api_distributed_is_initialized:

is_initialized
-------------------------------


.. py:function:: paddle.distributed.is_initialized()

检查默认通信组是否已经初始化

参数
:::::::::
无

返回
:::::::::
如果默认通信组已被初始化则返回True；反之则返回False。

代码示例
:::::::::
COPY-FROM: paddle.distributed.is_initialized