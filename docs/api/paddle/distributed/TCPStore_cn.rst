.. _cn_api_distributed_TCPStore:

TCPStore
-------------------------------

.. py:class:: paddle.distributed.TCPStore(host_name, port, word_size=1, is_master=False, time_out)

该OP是实现存储TCP的分布式键值，通过初始化服务器存储，客户端通过TCP连接到服务器可以完成set()插入键值对、get()返回键值对等一些操作。

**参数**
    - **host_name** (str)- 服务器的主机名或IP地址。
    - **port** (int)- 服务器监听请求的端口。
    - **world_size** (int，可选)- 总共使用的服务器和客户端数量。默认值是1。
    - **is_master** (bool，可选)- True表示初始化服务器和False表示初始化客户端。默认值为False。
    - **timeout** (timedelta，可选)- 初始化存储允许的最大超时时间。默认值是360s。

代码示例
::::::::::::

.. code-block:: python

    import datetime
    import paddle

    store = paddle.distributed.TCPStore("127.0.0.1", 6170, True, 1,
                                           datetime.timedelta(0))
    store.add("my", 3)
    ret = store.get('my')
    print(ret)


add(**kwargs)

该OP在第一次调用将创建一个与key关联的计数器，并初始化为value。后续调用使用相同的键计数器增加的数量。

**参数**
    - **key** (str)- 存储中计数器递增的key。
    - **value** (str)- 计数器增加的数值。


**代码示例**：

.. code-block:: python

    import paddle

    store = paddle.fluid.core.TCPStore("127.0.0.1", 6170, True, 1,
                                           datetime.timedelta(0))
    store.add("my", 3)
    store.add("my", 3)
    ret = store.get('my')
    print(ret)

get(**kwargs)

该OP返回给定key对应的值。

**参数**
    - **key** (str)- 在存储中给定的key。

**返回**
    返回在存储中给定key对应的数值。

**代码示例**：

.. code-block:: python

    import datetime
    import paddle

    store = paddle.fluid.core.TCPStore("127.0.0.1", 6170, True, 1,
                                           datetime.timedelta(0))
    store.add("my", 3)
    ret = store.get('my')
    print(ret)

set(**kwargs)

该OP是将根据输入的key-value插入存储，如果key已经在存储中存在，将用新提供的value覆盖旧的值。

**参数**
    - **key** (str)- 添加到存储中的key。
    - **value** (str)- 添加到存储中与key相关联的值。


**代码示例**：

.. code-block:: python

    import datetime
    import paddle

    store = paddle.fluid.core.TCPStore("127.0.0.1", 6170, True, 1,
                                           datetime.timedelta(0))
    store.set("key",3)
    ret = store.get("key")
    print(ret)

wait(**kwargs)

该OP对将key添加到存储超时(存储初始化时设置)的情况抛出异常。

**参数**
    - **key** (str)- 需要等待的key。


**代码示例**：

.. code-block:: python

    import datetime
    import paddle

    store = paddle.fluid.core.TCPStore("127.0.0.1", 6170, True, 1,
                                           datetime.timedelta(0))
    store.wait("my")





