#################
 fluid.io
#################



.. _cn_api_fluid_io_load_inference_model:

load_inference_model
-------------------------------

.. py:function:: paddle.fluid.io.load_inference_model(dirname, executor, model_filename=None, params_filename=None, pserver_endpoints=None)

从指定目录中加载预测模型(inference model)。通过这个API，您可以获得模型结构（预测程序）和模型参数。如果您只想下载预训练后的模型的参数，请使用load_params API。更多细节请参考 ``模型/变量的保存、载入与增量训练`` 。

参数:
  - **dirname** (str) – model的路径
  - **executor** (Executor) – 运行 inference model的 ``executor``
  - **model_filename** (str|None) –  存储着预测 Program 的文件名称。如果设置为None，将使用默认的文件名为： ``__model__``
  - **params_filename** (str|None) –  加载所有相关参数的文件名称。如果设置为None，则参数将保存在单独的文件中。
  - **pserver_endpoints** (list|None) – 只有在分布式预测时需要用到。 当在训练时使用分布式 look up table , 需要这个参数. 该参数是 pserver endpoints 的列表

返回: 这个函数的返回有三个元素的元组(Program，feed_target_names, fetch_targets)。Program 是一个 ``Program`` ，它是预测 ``Program``。  ``feed_target_names`` 是一个str列表，它包含需要在预测 ``Program`` 中提供数据的变量的名称。``fetch_targets`` 是一个 ``Variable`` 列表，从中我们可以得到推断结果。

返回类型：元组(tuple)

抛出异常：
   - ``ValueError`` – 如果 ``dirname`` 非法 

.. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        main_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
            w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32')
            b = fluid.layers.create_parameter(shape=[200], dtype='float32')
            hidden_w = fluid.layers.matmul(x=data, y=w)
            hidden_b = fluid.layers.elementwise_add(hidden_w, b)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_prog)
        path = "./infer_model"
        fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'],target_vars=[hidden_b], executor=exe, main_program=main_prog)
        tensor_img = np.array(np.random.random((1, 64, 784)), dtype=np.float32)
        [inference_program, feed_target_names, fetch_targets] = (fluid.io.load_inference_model(dirname=path, executor=exe))
        
        results = exe.run(inference_program,
                  feed={feed_target_names[0]: tensor_img},
                  fetch_list=fetch_targets)

        # endpoints是pserver服务器终端列表，下面仅为一个样例
        endpoints = ["127.0.0.1:2023","127.0.0.1:2024"]
        # 如果需要查询表格，我们可以使用：
        [dist_inference_program, dist_feed_target_names, dist_fetch_targets] = (
            fluid.io.load_inference_model(dirname=path,
                                          executor=exe,
                                          pserver_endpoints=endpoints))

        # 在这个示例中，inference program 保存在“ ./infer_model/__model__”中
        # 参数保存在“./infer_mode ”单独的若干文件中
        # 加载 inference program 后， executor 使用 fetch_targets 和 feed_target_names 执行Program，得到预测结果







.. _cn_api_fluid_io_load_params:

load_params
-------------------------------

.. py:function:: paddle.fluid.io.load_params(executor, dirname, main_program=None, filename=None)

该函数从给定 ``main_program`` 中取出所有参数，然后从目录 ``dirname`` 中或 ``filename`` 指定的文件中加载这些参数。

``dirname`` 用于存有变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用filename来指明这个文件。

注意:有些变量不是参数，但它们对于训练是必要的。因此，调用 ``save_params()`` 和 ``load_params()`` 来保存和加载参数是不够的，可以使用 ``save_persistables()`` 和 ``load_persistables()`` 代替这两个函数。

如果您想下载预训练后的模型结构和参数用于预测，请使用load_inference_model API。更多细节请参考 :ref:`api_guide_model_save_reader`。

参数:
    - **executor**  (Executor) – 加载变量的 executor
    - **dirname**  (str) – 目录路径
    - **main_program**  (Program|None) – 需要加载变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
    - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None

返回: None
  
**代码示例**

.. code-block:: python

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.load_params(executor=exe, dirname=param_path,
                        main_program=None)







.. _cn_api_fluid_io_load_persistables:

load_persistables
-------------------------------

.. py:function:: paddle.fluid.io.load_persistables(executor, dirname, main_program=None, filename=None)

该函数从给定 ``main_program`` 中取出所有 ``persistable==True`` 的变量（即长期变量），然后将它们从目录 ``dirname`` 中或 ``filename`` 指定的文件中加载出来。

``dirname`` 用于指定存有长期变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用filename来指定它。

参数:
    - **executor**  (Executor) – 加载变量的 executor
    - **dirname**  (str) – 目录路径
    - **main_program**  (Program|None) – 需要加载变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
    - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None

返回: None
  
**代码示例**

.. code-block:: python

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.load_persistables(executor=exe, dirname=param_path,
                               main_program=None)
 






.. _cn_api_fluid_io_load_vars:

load_vars
-------------------------------

.. py:function:: paddle.fluid.io.load_vars(executor, dirname, main_program=None, vars=None, predicate=None, filename=None)

``executor`` 从指定目录加载变量。

有两种方法来加载变量:方法一，``vars`` 为变量的列表。方法二，将已存在的 ``Program`` 赋值给 ``main_program`` ，然后将加载 ``Program`` 中的所有变量。第一种方法优先级更高。如果指定了 vars，那么忽略 ``main_program`` 和 ``predicate`` 。

``dirname`` 用于指定加载变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用 ``filename`` 来指定它。

参数:
 - **executor**  (Executor) – 加载变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要加载变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **vars**  (list[Variable]|None) –  要加载的变量的列表。 优先级高于main_program。默认值: None
 - **predicate**  (function|None) – 如果不等于None，当指定main_program， 那么只有 predicate(variable)==True 时，main_program中的变量会被加载。
 - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None

抛出异常：
  - ``TypeError`` - 如果参数 ``main_program`` 为 None 或为一个非 ``Program`` 的实例
   
返回: None
  
**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid
    main_prog = fluid.Program()
    startup_prog = fluid.Program()
    with fluid.program_guard(main_prog, startup_prog):
        data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
        w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
        b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
        hidden_w = fluid.layers.matmul(x=data, y=w)
        hidden_b = fluid.layers.elementwise_add(hidden_w, b)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    param_path = "./my_paddle_model"

    # 第一种使用方式 使用 main_program 指定变量
    def name_has_fc(var):
        res = "fc" in var.name
        return res
    fluid.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate=name_has_fc)
    fluid.io.load_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate=name_has_fc)
    #加载所有`main_program`中变量名包含 ‘fc’ 的变量
    #并且此前所有变量应该保存在不同文件中

    #用法2：使用 `vars` 来使变量具体化
    path = "./my_paddle_vars"
    var_list = [w, b]
    fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                       filename="vars_file")
    fluid.io.load_vars(executor=exe, dirname=path, vars=var_list,
                       filename="vars_file")
    # 加载w和b，它们此前应被保存在同一名为'var_file'的文件中
    # 该文件所在路径为 "./my_paddle_model"
 


.. _cn_api_fluid_io_PyReader:

PyReader
-------------------------------

.. py:class:: paddle.fluid.io.PyReader(feed_list=None, capacity=None, use_double_buffer=True, iterable=True, return_list=False)


在python中为数据输入创建一个reader对象。将使用python线程预取数据，并将其异步插入队列。当调用Executor.run（…）时，将自动提取队列中的数据。 

参数:
  - **feed_list** (list(Variable)|tuple(Variable))  – feed变量列表，由 ``fluid.layers.data()`` 创建。在可迭代模式下它可以被设置为None。
  - **capacity** (int) – 在Pyreader对象中维护的队列的容量。
  - **use_double_buffer** (bool) – 是否使用 ``double_buffer_reader`` 来加速数据输入。
  - **iterable** (bool) –  被创建的reader对象是否可迭代。
  - **eturn_list** (bool) –  是否以list的形式将返回值

返回: 被创建的reader对象

返回类型： reader (Reader)


**代码示例**

1.如果iterable=False，则创建的Pyreader对象几乎与 ``fluid.layers.py_reader（）`` 相同。算子将被插入program中。用户应该在每个epoch之前调用start（），并在epoch结束时捕获 ``Executor.run（）`` 抛出的 ``fluid.core.EOFException `` 。一旦捕获到异常，用户应该调用reset（）手动重置reader。

.. code-block:: python

    EPOCH_NUM = 3
    ITER_NUM = 5
    BATCH_SIZE = 3

    def reader_creator_random_image_and_label(height, width):
        def reader():
            for i in range(ITER_NUM):
                fake_image = np.random.uniform(low=0,
                                               high=255,
                                               size=[height, width])
                fake_label = np.ones([1])
                yield fake_image, fake_label
        return reader

    image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    reader = fluid.io.PyReader(feed_list=[image, label],
                               capacity=4,
                               iterable=False)

    user_defined_reader = reader_creator_random_image_and_label(784, 784)
    reader.decorate_sample_list_generator(
        paddle.batch(user_defined_reader, batch_size=BATCH_SIZE))
    # 此处省略网络定义
    executor = fluid.Executor(fluid.CUDAPlace(0))
    executor.run(fluid.default_startup_program())
    for i in range(EPOCH_NUM):
        reader.start()
        while True:
            try:
                executor.run(feed=None)
            except fluid.core.EOFException:
                reader.reset()
                break


2.如果iterable=True，则创建的Pyreader对象与程序分离。程序中不会插入任何算子。在本例中，创建的reader是一个python生成器，它是不可迭代的。用户应将从Pyreader对象生成的数据输入 ``Executor.run(feed=...)`` 。

.. code-block:: python

   EPOCH_NUM = 3
   ITER_NUM = 5
   BATCH_SIZE = 10

   def reader_creator_random_image(height, width):
       def reader():
           for i in range(ITER_NUM):
               yield np.random.uniform(low=0, high=255, size=[height, width]),
       return reader

   image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
   reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=True, return_list=False)

   user_defined_reader = reader_creator_random_image(784, 784)
   reader.decorate_sample_list_generator(
       paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
       fluid.core.CUDAPlace(0))
   # 此处省略网络定义
   executor = fluid.Executor(fluid.CUDAPlace(0))
   executor.run(fluid.default_main_program())

   for _ in range(EPOCH_NUM):
       for data in reader():
           executor.run(feed=data)

3. return_list=True，返回值将用list表示而非dict

.. code-block:: python

   import paddle
   import paddle.fluid as fluid
   import numpy as np

   EPOCH_NUM = 3
   ITER_NUM = 5
   BATCH_SIZE = 10

   def reader_creator_random_image(height, width):
       def reader():
           for i in range(ITER_NUM):
               yield np.random.uniform(low=0, high=255, size=[height, width]),
       return reader

   image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
   reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=True, return_list=True)

   user_defined_reader = reader_creator_random_image(784, 784)
   reader.decorate_sample_list_generator(
       paddle.batch(user_defined_reader, batch_size=BATCH_SIZE),
       fluid.core.CPUPlace())
   # 此处省略网络定义
   executor = fluid.Executor(fluid.core.CPUPlace())
   executor.run(fluid.default_main_program())

   for _ in range(EPOCH_NUM):
       for data in reader():
           executor.run(feed={"image": data[0]})



.. py:method:: start()

启动数据输入线程。只能在reader对象不可迭代时调用。

**代码示例**

.. code-block:: python

  BATCH_SIZE = 10
     
  def generator():
    for i in range(5):
       yield np.random.uniform(low=0, high=255, size=[784, 784]),
     
  image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
  reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
  reader.decorate_sample_list_generator(
    paddle.batch(generator, batch_size=BATCH_SIZE))
     
  executor = fluid.Executor(fluid.CUDAPlace(0))
  executor.run(fluid.default_startup_program())
  for i in range(3):
    reader.start()
    while True:
        try:
            executor.run(feed=None)
        except fluid.core.EOFException:
            reader.reset()
            break

.. py:method:: reset()

当 ``fluid.core.EOFException`` 抛出时重置reader对象。只能在reader对象不可迭代时调用。

**代码示例**

.. code-block:: python

            BATCH_SIZE = 10
     
            def generator():
                for i in range(5):
                    yield np.random.uniform(low=0, high=255, size=[784, 784]),
     
            image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
            reader = fluid.io.PyReader(feed_list=[image], capacity=4, iterable=False)
            reader.decorate_sample_list_generator(
                paddle.batch(generator, batch_size=BATCH_SIZE))
     
            executor = fluid.Executor(fluid.CUDAPlace(0))
            executor.run(fluid.default_startup_program())
            for i in range(3):
                reader.start()
                while True:
                    try:
                        executor.run(feed=None)
                    except fluid.core.EOFException:
                        reader.reset()
                        break

.. py:method:: decorate_sample_generator(sample_generator, batch_size, drop_last=True, places=None)

设置Pyreader对象的数据源。

提供的 ``sample_generator`` 应该是一个python生成器，它生成的数据类型应为list(numpy.ndarray)。

当Pyreader对象不可迭代时，必须设置 ``places`` 。

如果所有的输入都没有LOD，这个方法比 ``decorate_sample_list_generator(paddle.batch(sample_generator, ...))`` 更快。

参数:
  - **sample_generator** (generator)  – Python生成器，yield 类型为list(numpy.ndarray)
  - **batch_size** (int) – batch size，必须大于0
  - **drop_last** (bool) – 当样本数小于batch数量时，是否删除最后一个batch
  - **places** (None|list(CUDAPlace)|list(CPUPlace)) –  位置列表。当PyReader可迭代时必须被提供

**代码示例**

.. code-block:: python
     
            EPOCH_NUM = 3
            ITER_NUM = 15
            BATCH_SIZE = 3
     
            def random_image_and_label_generator(height, width):
                def generator():
                    for i in range(ITER_NUM):
                        fake_image = np.random.uniform(low=0,
                                                       high=255,
                                                       size=[height, width])
                        fake_label = np.array([1])
                        yield fake_image, fake_label
                return generator
     
            image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int32')
            reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)
     
            user_defined_generator = random_image_and_label_generator(784, 784)
            reader.decorate_sample_generator(user_defined_generator,
                                             batch_size=BATCH_SIZE,
                                             places=[fluid.CUDAPlace(0)])
            # 省略了网络的定义
            executor = fluid.Executor(fluid.CUDAPlace(0))
            executor.run(fluid.default_main_program())
     
            for _ in range(EPOCH_NUM):
                for data in reader():
                    executor.run(feed=data)

.. py:method:: decorate_sample_list_generator(reader, places=None)

设置Pyreader对象的数据源。

提供的 ``reader`` 应该是一个python生成器，它生成列表（numpy.ndarray）类型的批处理数据。

当Pyreader对象不可迭代时，必须设置 ``places`` 。

参数:
  - **reader** (generator)  – 返回列表（numpy.ndarray）类型的批处理数据的Python生成器
  - **places** (None|list(CUDAPlace)|list(CPUPlace)) –  位置列表。当PyReader可迭代时必须被提供

**代码示例**

.. code-block:: python
            
            EPOCH_NUM = 3
            ITER_NUM = 15
            BATCH_SIZE = 3
     
            def random_image_and_label_generator(height, width):
                def generator():
                    for i in range(ITER_NUM):
                        fake_image = np.random.uniform(low=0,
                                                       high=255,
                                                       size=[height, width])
                        fake_label = np.ones([1])
                        yield fake_image, fake_label
                return generator
     
            image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int32')
            reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)
     
            user_defined_generator = random_image_and_label_generator(784, 784)
            reader.decorate_sample_list_generator(
                paddle.batch(user_defined_generator, batch_size=BATCH_SIZE),
                fluid.core.CUDAPlace(0))
            # 省略了网络的定义
            executor = fluid.Executor(fluid.core.CUDAPlace(0))
            executor.run(fluid.default_main_program())
     
            for _ in range(EPOCH_NUM):
                for data in reader():
                    executor.run(feed=data)

.. py:method:: decorate_batch_generator(reader, places=None)

设置Pyreader对象的数据源。

提供的 ``reader`` 应该是一个python生成器，它生成列表（numpy.ndarray）类型或LoDTensor类型的批处理数据。

当Pyreader对象不可迭代时，必须设置 ``places`` 。

参数:
  - **reader** (generator)  – 返回LoDTensor类型的批处理数据的Python生成器
  - **places** (None|list(CUDAPlace)|list(CPUPlace)) –  位置列表。当PyReader可迭代时必须被提供

**代码示例**

.. code-block:: python

            EPOCH_NUM = 3
            ITER_NUM = 15
            BATCH_SIZE = 3
     
            def random_image_and_label_generator(height, width):
                def generator():
                    for i in range(ITER_NUM):
                        batch_image = np.random.uniform(low=0,
                                                        high=255,
                                                        size=[BATCH_SIZE, height, width])
                        batch_label = np.ones([BATCH_SIZE, 1])
                        yield batch_image, batch_label
                return generator
     
            image = fluid.layers.data(name='image', shape=[784, 784], dtype='float32')
            label = fluid.layers.data(name='label', shape=[1], dtype='int32')
            reader = fluid.io.PyReader(feed_list=[image, label], capacity=4, iterable=True)
     
            user_defined_generator = random_image_and_label_generator(784, 784)
            reader.decorate_batch_generator(user_defined_generator, fluid.CUDAPlace(0))
            # 省略了网络的定义
            executor = fluid.Executor(fluid.CUDAPlace(0))
            executor.run(fluid.default_main_program())
     
            for _ in range(EPOCH_NUM):
                for data in reader():
                    executor.run(feed=data)


.. _cn_api_fluid_io_save_inference_model:

save_inference_model
-------------------------------

.. py:function:: paddle.fluid.io.save_inference_model(dirname, feeded_var_names, target_vars, executor, main_program=None, model_filename=None, params_filename=None, export_for_deployment=True,  program_only=False)

修改指定的 ``main_program`` ，构建一个专门用于预测的 ``Program``，然后  ``executor`` 把它和所有相关参数保存到 ``dirname`` 中。


``dirname`` 用于指定保存变量的目录。如果变量保存在指定目录的若干文件中，设置文件名 None; 如果所有变量保存在一个文件中，请使用filename来指定它。

如果您仅想保存您训练好的模型的参数，请使用save_params API。更多细节请参考 :ref:`api_guide_model_save_reader` 。


参数:
  - **dirname** (str) – 保存预测model的路径
  - **feeded_var_names** (list[str]) – 预测（inference）需要 feed 的数据
  - **target_vars** (list[Variable]) – 保存预测（inference）结果的 Variables
  - **executor** (Executor) –  executor 保存  inference model
  - **main_program** (Program|None) – 使用 ``main_program`` ，构建一个专门用于预测的 ``Program`` （inference model）. 如果为None, 使用   ``default main program``   默认: None.
  - **model_filename** (str|None) – 保存预测Program 的文件名称。如果设置为None，将使用默认的文件名为： ``__model__``
  - **params_filename** (str|None) – 保存所有相关参数的文件名称。如果设置为None，则参数将保存在单独的文件中。
  - **export_for_deployment** (bool) – 如果为真，Program将被修改为只支持直接预测部署的Program。否则，将存储更多的信息，方便优化和再训练。目前只支持True。
  - **program_only** (bool) – 如果为真，将只保存预测程序，而不保存程序的参数。

返回: 获取的变量名列表

返回类型：target_var_name_list(list)

抛出异常：
 - ``ValueError`` – 如果 ``feed_var_names`` 不是字符串列表
 - ``ValueError`` – 如果 ``target_vars`` 不是 ``Variable`` 列表

**代码示例**

.. code-block:: python

    import paddle.fluid as fluid

    path = "./infer_model"

    # 用户定义网络，此处以softmax回归为例
    image = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
    predict = fluid.layers.fc(input=image, size=10, act='softmax')

    loss = fluid.layers.cross_entropy(input=predict, label=label)
    avg_loss = fluid.layers.mean(loss)

    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    # 数据输入及训练过程

    # 保存预测模型。注意我们不在这个示例中保存标签和损失。
    fluid.io.save_inference_model(dirname=path, feeded_var_names=['img'], target_vars=[predict], executor=exe)

    # 在这个示例中，函数将修改默认的主程序让它适合于预测‘predict_var’
    # 修改的预测Program 将被保存在 ./infer_model/__model__”中。
    # 参数将保存在文件夹下的单独文件中 ./infer_mode








.. _cn_api_fluid_io_save_params:

save_params
-------------------------------

.. py:function:: paddle.fluid.io.save_params(executor, dirname, main_program=None, filename=None)

该函数从 ``main_program`` 中取出所有参数，然后将它们保存到 ``dirname`` 目录下或名为 ``filename`` 的文件中。

``dirname`` 用于指定保存变量的目标目录。如果想将变量保存到多个独立文件中，设置 ``filename`` 为 None; 如果想将所有变量保存在单个文件中，请使用 ``filename`` 来指定该文件的命名。

注意:有些变量不是参数，但它们对于训练是必要的。因此，调用 ``save_params()`` 和 ``load_params()`` 来保存和加载参数是不够的，可以使用 ``save_persistables()`` 和 ``load_persistables()`` 代替这两个函数。如果您想要储存您的模型用于预测，请使用save_inference_model API。更多细节请参考 :ref:`api_guide_model_save_reader`。


参数:
 - **executor**  (Executor) – 保存变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要保存变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **vars**  (list[Variable]|None) –  要保存的所有变量的列表。 优先级高于main_program。默认值: None
 - **filename**  (str|None) – 保存变量的文件。如果想分不同独立文件来保存变量，设置 filename=None. 默认值: None
 
返回: None
  
**代码示例**

.. code-block:: python
    
    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.save_params(executor=exe, dirname=param_path,
                         main_program=None)
                         






.. _cn_api_fluid_io_save_persistables:

save_persistables
-------------------------------

.. py:function:: paddle.fluid.io.save_persistables(executor, dirname, main_program=None, filename=None)

该函数从给定 ``main_program`` 中取出所有 ``persistable==True`` 的变量，然后将它们保存到目录 ``dirname`` 中或 ``filename`` 指定的文件中。

``dirname`` 用于指定保存长期变量的目录。如果想将变量保存到指定目录的若干文件中，设置 ``filename=None`` ; 如果想将所有变量保存在一个文件中，请使用 ``filename`` 来指定它。

参数:
 - **executor**  (Executor) – 保存变量的 executor
 - **dirname**  (str) – 目录路径
 - **main_program**  (Program|None) – 需要保存变量的 Program。如果为 None，则使用 default_main_Program 。默认值: None
 - **predicate**  (function|None) – 如果不等于None，当指定main_program， 那么只有 predicate(variable)==True 时，main_program中的变量
 - **vars**  (list[Variable]|None) –  要保存的所有变量的列表。 优先级高于main_program。默认值: None
 - **filename**  (str|None) – 保存变量的文件。如果想分开保存变量，设置 filename=None. 默认值: None
 
返回: None
  
**代码示例**

.. code-block:: python
    
    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    # `prog` 可以是由用户自定义的program
    fluid.io.save_persistables(executor=exe, dirname=param_path,
                               main_program=prog)
    
    






.. _cn_api_fluid_io_save_vars:

save_vars
-------------------------------

.. py:function:: paddle.fluid.io.save_vars(executor, dirname, main_program=None, vars=None, predicate=None, filename=None)

通过 ``Executor`` ,此函数将变量保存到指定目录下。

有两种方法可以指定要保存的变量：第一种方法，在列表中列出变量并将其传给 ``vars`` 参数。第二种方法是，将现有程序分配给 ``main_program`` ，它会保存program中的所有变量。第一种方式具有更高的优先级。换句话说，如果分配了变量，则将忽略 ``main_program`` 和 ``predicate`` 。

``dirname`` 用于指定保存变量的文件夹。如果您希望将变量分别保存在文件夹目录的多个单独文件中，请设置 ``filename`` 为无；如果您希望将所有变量保存在单个文件中，请使用 ``filename`` 指定它。

参数：
      - **executor** （Executor）- 为保存变量而运行的执行器。
      - **dirname** （str）- 目录路径。
      - **main_program** （Program | None）- 保存变量的程序。如果为None，将自动使用默认主程序。默认值：None。
      - **vars** （list [Variable] | None）- 包含要保存的所有变量的列表。它的优先级高于 ``main_program`` 。默认值：None。
      - **predicate** （function | None）- 如果它不是None，则只保存 ``main_program`` 中使 :math:`predicate(variable)== True` 的变量。它仅在我们使用 ``main_program`` 指定变量时才起作用（换句话说，vars为None）。默认值：None。
      - **filename** （str | None）- 保存所有变量的文件。如果您希望单独保存变量，请将其设置为None。默认值：None。

返回：     None

抛出异常：    
    - ``TypeError`` - 如果main_program不是Program的实例，也不是None。

**代码示例**

.. code-block:: python
      
      import paddle.fluid as fluid
      main_prog = fluid.Program()
      startup_prog = fluid.Program()
      with fluid.program_guard(main_prog, startup_prog):
          data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
          w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
          b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
          hidden_w = fluid.layers.matmul(x=data, y=w)
          hidden_b = fluid.layers.elementwise_add(hidden_w, b)
      place = fluid.CPUPlace()
      exe = fluid.Executor(place)
      exe.run(startup_prog)
     
      param_path = "./my_paddle_model"

      # 第一种用法:用main_program来指定变量。
      def name_has_fc(var):
          res = "fc" in var.name
          return res

      fluid.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate = name_has_fc)
      # 将main_program中名中包含“fc”的的所有变量保存。
      # 变量将分开保存。


      # 第二种用法: 用vars来指定变量。
      var_list = [w, b]
      path = "./my_paddle_vars"
      fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                         filename="vars_file")
      # var_a，var_b和var_c将被保存。
      #他们将使用同一文件，名为“var_file”，保存在路径“./my_paddle_vars”下。






