.. _cn_api_distributed_fleet_DataGenerator:

DataGenerator
-------------------------------


.. py:class:: paddle.distributed.fleet.DataGenerator



DataGenerator是一个基类，用户在使用InMemoryDataset/QueueDataset时，如果希望实现自己的python数据预处理逻辑，可以继承DataGenerator类


.. py:method:: set_batch(batch_size)

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

设置当前DataGenerator的batch size，仅在用户定义generate_batch时生效。

参数：
    - **batch_size** - batch size 的大小。

返回：None。


**代码示例**

.. code-block:: python


    import paddle.distributed.fleet.data_generator as dg
    class MyData(dg.DataGenerator):

        def generate_sample(self, line):
            def local_iter():
                int_words = [int(x) for x in line.split()]
                yield ("words", int_words)
            return local_iter

        def generate_batch(self, samples):
            def local_iter():
                for s in samples:
                    yield ("words", s[1].extend([s[1][0]]))
    mydata = MyData()
    mydata.set_batch(128)

.. py:method:: run_from_memory()

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

从内存中生成数据，通常用于调试。

参数：None

返回：None。


**代码示例**

.. code-block:: python

    import paddle.distributed.fleet.data_generator as dg
    class MyData(dg.DataGenerator):

        def generate_sample(self, line):
            def local_iter():
                yield ("words", [1, 2, 3, 4])
            return local_iter

    mydata = MyData()
    mydata.run_from_memory()


.. py:method:: run_from_stdin()

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

从stdin中读取原始数据经过generate_sample函数之后的数据，并进一步通过_gen_str()方法，经过处理后的数据输出到stdout

参数：None

返回：None。


**代码示例**

.. code-block:: python

    import paddle.distributed.fleet.data_generator as dg
    class MyData(dg.DataGenerator):

        def generate_sample(self, line):
            def local_iter():
                int_words = [int(x) for x in line.split()]
                yield ("words", [int_words])
            return local_iter

    mydata = MyData()
    mydata.run_from_stdin()


.. py:method:: _gen_str(line)

处理generate_sample的结果，需要子类实现



参数：
    - **line** (str) - 又用户自定义的generate_samle()函数的输出


.. py:method:: generate_sample(line)

**注意：**

  **1. 该API只在非** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **模式下生效**

需要被子类定义，用于处理原始训练数据到list或者tuple

参数：
    - **line** (str) - 一条原始训练数据。

返回：返回自定义处理逻辑之后的数据，格式为list或者tuple：[(name, [feasign, ...]), ...] 或者 ((name, [feasign, ...]), ...)
例如：[("words", [1926, 08, 17]), ("label", [1])] 或者 (("words", [1926, 08, 17]), ("label", [1]))

**代码示例**:

.. code-block:: python

    import paddle.distributed.fleet.data_generator as dg
    class MyData(dg.DataGenerator):

        def generate_sample(self, line):
            def local_iter():
                int_words = [int(x) for x in line.split()]
                yield ("words", [int_words])
            return local_iter

.. py:method:: generate_batch(samples)

需要子类自定义，处理由generate_sample(line)生成的一个batch的数据的处理逻辑，一般需要对一个batch的数据进行处理时需要自定义该函数。
例如：根据一个batch内样本的最大长度做padding。

参数：
    - **samples** (list | tuple) - 由generate_sample(line)生成的一个batch。

返回：一个python生成器，和generate_sample(line)生成的数据同一格式。
**代码示例**:

.. code-block:: python

    import paddle.distributed.fleet.data_generator as dg
    class MyData(dg.DataGenerator):

        def generate_sample(self, line):
            def local_iter():
                int_words = [int(x) for x in line.split()]
                yield ("words", int_words)
            return local_iter

        def generate_batch(self, samples):
            def local_iter():
                for s in samples:
                    yield ("words", s[1].extend([s[1][0]]))
    mydata = MyData()
    mydata.set_batch(128)