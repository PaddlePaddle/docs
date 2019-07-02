#################
Data Reader
#################


.. _cn_api_paddle_data_reader_datafeeder:

DataFeeder
==================================

.. py:class:: paddle.fluid.data_feeder.DataFeeder(feed_list, place, program=None)


DataFeeder将reader返回的数据转换为可以输入Executor和ParallelExecutor的数据结构。reader通常返回一个小批量数据条目列表。列表中的每个数据条目都是一个样本。每个样本都是具有一个或多个特征的列表或元组。

简单用法如下：

**代码示例**

..  code-block:: python

    import paddle.fluid as fluid
    place = fluid.CPUPlace()
    img = fluid.layers.data(name='image', shape=[1, 28, 28])
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
    result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])


如果您想在使用多个GPU训练模型时预先将数据单独输入GPU端，可以使用decorate_reader函数。


**代码示例**

..  code-block:: python

    import paddle
    import paddle.fluid as fluid
    
    place=fluid.CUDAPlace(0)
    data = fluid.layers.data(name='data', shape=[3, 224, 224], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    
    feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
    reader = feeder.decorate_reader(
        paddle.batch(paddle.dataset.flowers.train(), batch_size=16), multi_devices=False)


参数：
    - **feed_list**  (list) –  将输入模型的变量或变量的名称。
    - **place**  (Place) – place表示将数据输入CPU或GPU，如果要将数据输入GPU，请使用fluid.CUDAPlace(i)（i表示GPU的ID），如果要将数据输入CPU，请使用fluid.CPUPlace()。
    - **program**  (Program) –将数据输入的Program，如果Program为None，它将使用default_main_program() 。默认值None。

抛出异常：     ``ValueError`` – 如果某些变量未在Program中出现


**代码示例**

..  code-block:: python

    import numpy as np
    import paddle
    import paddle.fluid as fluid

    place = fluid.CPUPlace()

    def reader():
        yield [np.random.random([4]).astype('float32'), np.random.random([3]).astype('float32')],

    main_program = fluid.Program()
    startup_program = fluid.Program()

    with fluid.program_guard(main_program, startup_program):
        data_1 = fluid.layers.data(name='data_1', shape=[1, 2, 2])
        data_2 = fluid.layers.data(name='data_2', shape=[1, 1, 3])
        out = fluid.layers.fc(input=[data_1, data_2], size=2)
        # ...

    feeder = fluid.DataFeeder([data_1, data_2], place)

    exe = fluid.Executor(place)
    exe.run(startup_program)
    for data in reader():
        outs = exe.run(program=main_program,
                       feed=feeder.feed(data),
                       fetch_list=[out])


.. py:method::  feed(iterable)

根据feed_list和iterable，将输入转换成一个数据结构，该数据结构可以输入Executor和ParallelExecutor。

参数：
    - **iterable** (list|tuple) – 输入的数据

返回： 转换结果

返回类型： dict

**代码示例**

..  code-block:: python

        import numpy.random as random
        import paddle.fluid as fluid
        
        def reader(limit=5):
            for i in range(limit):
                    yield random.random([784]).astype('float32'), random.random([1]).astype('int64'), random.random([256]).astype('float32')
        
        data_1 = fluid.layers.data(name='data_1', shape=[1, 28, 28])
        data_2 = fluid.layers.data(name='data_2', shape=[1], dtype='int64')
        data_3 = fluid.layers.data(name='data_3', shape=[16, 16], dtype='float32')
        feeder = fluid.DataFeeder(['data_1','data_2', 'data_3'], fluid.CPUPlace())
        
        result = feeder.feed(reader())



.. py:method::  feed_parallel(iterable, num_places=None)

需要多个mini-batches。每个mini-batch都将提前在每个设备上输入。

参数：
    - **iterable** (list|tuple) – 输入的数据。
    - **num_places**  (int) – 设备编号，默认值为None。

返回： 转换结果

返回类型： dict



.. note::

    设备数量和mini-batches数量必须一致。

**代码示例**

..  code-block:: python

        import numpy.random as random
        import paddle.fluid as fluid
        
        def reader(limit=10):
            for i in range(limit):
                yield [random.random([784]).astype('float32'), random.randint(10)],
        
        x = fluid.layers.data(name='x', shape=[1, 28, 28])
        y = fluid.layers.data(name='y', shape=[1], dtype='int64')
        
        feeder = fluid.DataFeeder(['x','y'], fluid.CPUPlace())
        place_num = 2
        places = [fluid.CPUPlace() for x in range(place_num)]
        data = []
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        program = fluid.CompiledProgram(fluid.default_main_program()).with_data_parallel(places=places)
        for item in reader():
            data.append(item)
            if place_num == len(data):
                exe.run(program=program, feed=list(feeder.feed_parallel(data, place_num)), fetch_list=[])
                data = []


.. py:method::  decorate_reader(reader, multi_devices, num_places=None, drop_last=True)

将输入数据转换成reader返回的多个mini-batches。每个mini-batch分别送入各设备中。

参数：
    - **reader** (function) – reader是可以生成数据的函数。
    - **multi_devices** (bool) – 是否用多个设备。
    - **num_places** (int) – 如果multi_devices是True, 你可以指定GPU的使用数量, 如果multi_devices是None, 会使用当前机器的所有GPU ，默认值None。
    - **drop_last** (bool) – 如果最后一个batch的大小小于batch_size，选择是否删除最后一个batch，默认值True。

返回： 转换结果

返回类型： dict

抛出异常：     ``ValueError`` – 如果drop_last为False并且数据batch和设备数目不匹配。

**代码示例**

..  code-block:: python

        import numpy.random as random
        import paddle
        import paddle.fluid as fluid
        
        def reader(limit=5):
            for i in range(limit):
                yield (random.random([784]).astype('float32'), random.random([1]).astype('int64')),
        
        place=fluid.CUDAPlace(0)
        data = fluid.layers.data(name='data', shape=[1, 28, 28], dtype='float32')
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        
        feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
        reader = feeder.decorate_reader(reader, multi_devices=False)
        
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        for data in reader():
            exe.run(feed=data)



.. _cn_api_paddle_data_reader_reader:

Reader
==================================

在训练和测试时，PaddlePaddle需要读取数据。为了简化用户编写数据读取代码的工作，我们定义了

    - reader是一个读取数据（从文件、网络、随机数生成器等）并生成数据项的函数。
    - reader creator是返回reader函数的函数。
    - reader decorator是一个函数，它接受一个或多个reader，并返回一个reader。
    - batch reader是一个函数，它读取数据（从reader、文件、网络、随机数生成器等）并生成一批数据项。


Data Reader Interface
------------------------------------

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

抛出异常：     ``ComposeNotAligned`` – reader的输出不一致。 当check_alignment设置为False，不会升高。



.. py:function:: paddle.reader.chain(*readers)

创建一个数据reader，其输出是链接在一起的输入数据reader的输出。

如果输入reader输出以下数据条目：[0，0，0][1，1，1][2，2，2]，链接reader将输出：[0，0，0，1，1，1，2，2，2] 。

参数：
    - **readers** – 输入的数据。

返回： 新的数据读取器

返回类型： callable


.. py:function:: paddle.reader.shuffle(reader, buf_size)

创建数据读取器，该reader的数据输出将被无序排列。

由原始reader创建的迭代器的输出将被缓冲到shuffle缓冲区，然后进行打乱。打乱缓冲区的大小由参数buf_size决定。

参数：
    - **reader** (callable)  – 输出会被打乱的原始reader
    - **buf_size** (int)  – 打乱缓冲器的大小

返回： 输出会被打乱的reader

返回类型： callable



.. py:function:: paddle.reader.firstn(reader, n)

限制reader可以返回的最大样本数。

参数：
    - **reader** (callable)  – 要读取的数据读取器。
    - **n** (int)  – 返回的最大样本数 。

返回： 装饰reader

返回类型： callable




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

param cut_lines:
     cut buffer to lines

type cut_lines:    bool

param line_break:
     line break of the file, like

or

type line_break:
     string

return:    one line or a buffer of bytes

rtype:    string



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



.. py:class::paddle.reader.Fake

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
