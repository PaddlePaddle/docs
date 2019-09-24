.. _cn_api_paddle_data_reader_reader:

Reader
-------------------------------------

在训练和测试时，PaddlePaddle需要读取数据。为了简化用户编写数据读取代码的工作，我们定义了

    - reader是一个读取数据（从文件、网络、随机数生成器等）并生成数据项的函数。
    - reader creator是返回reader函数的函数。
    - reader decorator是一个函数，它接受一个或多个reader，并返回一个reader。
    - batch reader是一个函数，它读取数据（从reader、文件、网络、随机数生成器等）并生成一批数据项。


Data Reader Interface
======================================

的确，data reader不必是读取和生成数据项的函数，它可以是任何不带参数的函数来创建一个iterable（任何东西都可以被用于 ``for x in iterable`` ):

..  code-block:: python

    iterable = data_reader()

从iterable生成的元素应该是单个数据条目，而不是mini batch。数据输入可以是单个项目，也可以是项目的元组，但应为 :ref:`user_guide_paddle_support_data_types` （如, numpy 1d array of float32, int, list of int）


单项目数据读取器创建者的示例实现：

..  code-block:: python

    def reader_creator_random_image(width, height):
        def reader():
            while True:
                yield numpy.random.uniform(-1, 1, size=width*height)
        return reader


多项目数据读取器创建者的示例实现：

..  code-block:: python

    def reader_creator_random_image_and_label(width, height, label):
        def reader():
            while True:
                yield numpy.random.uniform(-1, 1, size=width*height), label
        return reader

.. py:function::   paddle.reader.map_readers(func, *readers)

创建使用每个数据读取器的输出作为参数输出函数返回值的数据读取器。

参数：
    - **func**  - 使用的函数. 函数类型应为(Sample) => Sample
    - **readers**  - 其输出将用作func参数的reader。

类型：callable

返回： 被创建数据的读取器

返回类型： callable


.. py:function::  paddle.reader.buffered(reader, size)

创建缓冲数据读取器。

缓冲数据reader将读取数据条目并将其保存到缓冲区中。只要缓冲区不为空，就将继续从缓冲数据读取器读取数据。

参数：
    - **reader** (callable) - 要读取的数据读取器
    - **size** (int) - 最大缓冲


返回：缓冲数据的读取器


.. py:function::   paddle.reader.compose(*readers, **kwargs)

创建一个数据reader，其输出是输入reader的组合。

如果输入reader输出以下数据项：（1，2）3（4，5），则组合reader将输出：（1，2，3，4，5）。

参数：
    - **readers** - 将被组合的多个读取器。
    - **check_alignment** (bool) - 如果为True，将检查输入reader是否正确对齐。如果为False，将不检查对齐，将丢弃跟踪输出。默认值True。

返回：新的数据读取器

抛出异常：     ``ComposeNotAligned`` – reader的输出不一致。 当check_alignment设置为False，不会抛出异常。



.. py:function:: paddle.reader.chain(*readers)

创建一个数据reader，其输出是链接在一起的输入数据reader的输出。

如果输入reader输出以下数据条目：[0，0，0][1，1，1][2，2，2]，链接reader将输出：[0，0，0，1，1，1，2，2，2] 。

参数：
    - **readers** – 输入的数据。

返回： 新的数据读取器

返回类型： callable


.. py:function:: paddle.reader.shuffle(reader, buf_size)

该接口创建一个数据读取器，其功能是将原始数据读取器的数据打乱，然后返回无序的数据。

从原始数据读取器取出buf_size个数据到缓冲区，将缓冲区数据打乱，然后将无序的数据依次返回。当缓冲区数据全部输出后，再次执行上述步骤。

``paddle.reader.shuffle`` 是 ``paddle.fluid.io.shuffle`` 的别名。

参数：
    - **reader** (callable)  – 原始数据读取器。
    - **buf_size** (int)  – 缓冲区保存数据的个数。

返回： 返回无序数据的数据读取器

返回类型： callable

..  code-block:: python

    import paddle
    def reader():
        for i in range(5):
            yield i
    shuffled_reader = paddle.reader.shuffle(reader, 3)
    for e in shuffled_reader():
        print(e)
    # 输出结果是0~4的无序排列


.. py:function:: paddle.reader.firstn(reader, n)

该接口创建一个数据读取器，它可以返回的最大样本数为n。

``paddle.reader.firstn`` 是 ``paddle.fluid.io.firstn`` 的别名。

参数：
    - **reader** (callable)  – 输入的数据读取器。
    - **n** (int)  – 可以返回的最大样本数。

返回： 新的的数据读取器。

返回类型： callable

..  code-block:: python

    import paddle
    def reader():
        for i in range(100):
            yield i
    firstn_reader = paddle.reader.firstn(reader, 5)
    for e in firstn_reader():
        print(e)
    # 输出结果为:0 1 2 3 4 

.. py:function:: paddle.reader.xmap_readers(mapper, reader, process_num, buffer_size, order=False)

通过多线程方式，通过用户自定义的映射器mapper来映射reader返回的样本（到输出队列）。

参数：
    - **mapper** （callable） - 一种映射reader数据的函数。
    - **reader** （callable） - 产生数据的reader。
    - **process_num** （int） - 用于处理样本的线程数目。
    - **buffer_size** （int） - 存有待读取数据的队列的大小。
    - **order** （bool） - 是否保持原始reader的数据顺序。 默认为False。

返回：一个将原数据进行映射后的decorated reader。

返回类型： callable

.. py:class:: paddle.reader.PipeReader(command, bufsize=8192, file_type='plain')


PipeReader通过流从一个命令中读取数据，将它的stdout放到管道缓冲区中，并将其重定向到解析器进行解析，然后根据需要的格式生成数据。


您可以使用标准Linux命令或调用其他Program来读取数据，例如通过HDFS、CEPH、URL、AWS S3中读取：

**代码示例**

..  code-block:: python

    def example_reader():
        for f in myfiles:
            pr = PipeReader("cat %s"%f)
            for l in pr.get_line():
                sample = l.split(" ")
                yield sample


.. py:method:: get_line(cut_lines=True, line_break='\n')


参数：
    - **cut_lines** （bool） - 将缓冲区分行。
    - **line_break** （string） - 文件中的行分割符，比如 ‘\\n’ 或者 ‘\\r’。


返回：一行或者一段缓冲区。

返回类型： string



.. py:function:: paddle.reader.multiprocess_reader(readers, use_pipe=True, queue_size=1000)

多进程reader使用python多进程从reader中读取数据，然后使用multi process.queue或multi process.pipe合并所有数据。进程号等于输入reader的编号，每个进程调用一个reader。

multiprocess.queue需要/dev/shm的rw访问权限，某些平台不支持。

您需要首先创建多个reader，这些reader应该相互独立，这样每个进程都可以独立工作。

**代码示例**

..  code-block:: python

    reader0 = reader(["file01", "file02"])
    reader1 = reader(["file11", "file12"])
    reader1 = reader(["file21", "file22"])
    reader = multiprocess_reader([reader0, reader1, reader2],
        queue_size=100, use_pipe=False)



.. py:class:: paddle.reader.Fake

Fakereader将缓存它读取的第一个数据，并将其输出data_num次。它用于缓存来自真实reader的数据，并将其用于速度测试。

参数：
    - **reader** – 原始读取器。
    - **data_num** – reader产生数据的次数 。

返回： 一个Fake读取器


**代码示例**

..  code-block:: python

    def reader():
        for i in range(10):
            yield i

    fake_reader = Fake()(reader, 100)


Creator包包含一些简单的reader creator，可以在用户Program中使用。



.. py:function:: paddle.reader.creator.np_array(x)

如果是numpy向量，则创建一个生成x个元素的读取器。或者，如果它是一个numpy矩阵，创建一个生成x行元素的读取器。或由最高维度索引的任何子超平面。

参数：
    - **x** – 用于创建reader的numpy数组。

返回： 从x创建的数据读取器


.. py:function:: paddle.reader.creator.text_file(path)

创建从给定文本文件逐行输出文本的数据读取器。将删除每行的行尾的(‘\n’)。

路径：文本文件的路径

返回： 文本文件的数据读取器


.. py:function::  paddle.reader.creator.recordio(paths, buf_size=100)

从给定的recordio文件路径创建数据reader，用“，”分隔“，支持全局模式。

路径：recordio文件的路径，可以是字符串或字符串列表。

返回：recordio文件的数据读取器
