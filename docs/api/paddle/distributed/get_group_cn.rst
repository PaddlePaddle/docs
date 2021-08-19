.. _cn_api_distributed_get_group:

get_group
-------------------------------

.. py:function:: get_group(id=0)

通过通信组 id 获取通信组实例

参数
:::::::::
    - id (int): 通信组 id. 默认值为 0.

返回
:::::::::
Group 通信组实例

代码示例
:::::::::
.. code-block:: python

        ...
        gid = paddle.distributed.new_group([2,4,6])
        paddle.distributed.get_group(gid.id)

