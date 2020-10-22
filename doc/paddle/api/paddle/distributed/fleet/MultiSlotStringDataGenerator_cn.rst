.. _cn_api_distributed_fleet_MultiSlotStringDataGenerator_cn:

MultiSlotStringDataGenerator_cn
-------------------------------


.. py:class:: paddle.distributed.fleet.MultiSlotStringDataGenerator_cn

DataGenerator的子类实现，自定义了_gen_str(line)函数。与MultiSlotDataGenerator相比在python端直接处理string格式的数据，一般会有速度上的优势

.. py:method:: _gen_str(line)

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

进一步处理用户自定义的generate_sample(line)的处理逻辑。
输入一行数据的格式：
[(name, [str(feasign), ...]), ...] 或者 ((name, [str(feasign), ...]), ...)
输出格式：
[ids_num id1 id2 ...] ...
例如如果输入是这样：
[("words", ["1926", "08", "17"]), ("label", ["1"])]
输出：
3 1234 2345 3456 1 1

参数：
    - **line(str)** - 由用户自定义generate_sample(line)函数的输出。

返回：能够被C++端处理的字符串数据。

