.. _cn_api_io_map:

map
-------------------------------

.. py:class:: paddle.io.map(map_func, *args, **kwargs)

用于在GPU DataLoader流水线中划分数据预处理的阶段，每个阶段会通过独立的CUDA流和子线程来执行

参数
::::::::::::

    - **map_func** (callable) - 定义阶段内数据预处理的函数。
    - **args** - 传递给 ``map_func`` 的参数。
    - **kwargs** - 传递给 ``map_func`` 的关键字参数。

返回
::::::::::::
    数据预处理函数的输出


代码示例
::::::::::::

.. code-block:: python

COPY-FROM: <paddle.io.map>:<code-example>
