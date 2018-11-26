..  _api_guide_cluster_train_data:

####################
分布式训练数据准备
####################

一个数据并行的分布式训练任务通常会含有多个训练节点，每个训练节点负责训练整个数据集种的一部分。所以在
启动分布式训练任务之前需要将训练数据切分成多个小文件，通过一个 file_dispatcher 函数根据当前节点的
唯一序号(trainer_id)以及当前训练任务中训练节点的总数(trainers)决定读取哪一部分训练数据。

准备文本格式的分布式训练数据集
------------------------------

训练数据切分
~~~~~~~~~~~~

简单的，对于文本类训练数据来说，我们可以使用 split 命令将训练数据切分成多个小文件，例如：

  .. code-block:: bash
    $ split -d -a 4 -d -l 100 housing.data cluster/housing.data.
    $ find ./cluster
    cluster/
    cluster/housing.data.0002
    cluster/housing.data.0003
    cluster/housing.data.0004
    cluster/housing.data.0000
    cluster/housing.data.0001
    cluster/housing.data.0005

读取分布式训练数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在数据并行场景下，我们需要将训练数据平均分配给每个训练节点，通常的方法是实现一个函数，使之能够
根据当前任务的训练节点数量以及当前节点的唯一序号决定需要读取哪些文件，例如：

  .. code-block:: python

    def file_dispatcher(file_pattern, trainers, trainer_id):
      file_list = glob.glob(file_pattern)
      ret_list = []
      for idx, f in enumerate(file_list):
          if (idx + trainers) % trainers == trainer_id:
              ret_list.append(f)
      return ret_list

- file_pattern: 训练数据文件目录目录，上述例子可以是 `cluster/housing.data.*`
- trainers: 当前任务的训练节点数。
- trainer_id: 当前训练节点的唯一序号。

准备 RecordIO 格式的分布式训练数据集
-------------------------------------

对于非文本类数据，可以预先将训练数据转换为 RecordIO 格式再进行训练, 并且转换成 RecordIO 格式
的另一个好处是可以提升 IO 效率，从而提升分布式训练任务的运行效率。


生成 RecordIO 格式数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fluid 提供了 `fluid.recordio_writer.convert_reader_to_recordio_files` API, 可以将训练数据转换成
RecordIO 格式, 样例代码如下

  .. code-block:: python

      reader = paddle.batch(mnist.train(), batch_size=1)
      feeder = fluid.DataFeeder(
          feed_list=[  # order is image and label
              fluid.layers.data(
              name='image', shape=[784]),
              fluid.layers.data(
              name='label', shape=[1], dtype='int64'),
          ],
          place=fluid.CPUPlace())
      fluid.recordio_writer.convert_reader_to_recordio_files(
            filename_suffix='./mnist.recordio', batch_per_file=100, reader, feeder)

运行上述代码将会生成以下文件：

  .. code-block:: bash

      .
      \_mnist-00000.recordio
      |-mnist-00001.recordio
      |-mnist-00002.recordio
      |-mnist-00003.recordio
      |-mnist-00004.recordio

API Reference 请参考：:ref:`api_fluid_recordio_writer_convert_reader_to_recordio_file`

读取 RecordIO 训练数据
~~~~~~~~~~~~~~~~~~~~~~~~

Fluid 种提供了 `fluid.layers.io.open_files` API 来读取 RecordIO 格式的训练数据，在以下样例代码
中复用了上面例子中 `file_dispatcher` 函数来决定当前节点应该读取哪一部分训练数据：

  .. code-block:: python

    trainers = int(os.getenv("PADDLE_TRAINERS"))
    trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
    data_file = fluid.layers.io.open_files(
        filenames=file_dispatcher("./mnist-[0-9]*.recordio", 2, 0),
        thread_num=1,
        shapes=[(-1, 784),(-1, 1)],
        lod_levels=[0, 0],
        dtypes=["float32", "int32"])
    img, label = fluid.layers.io.read_file(data_files)

API Reference 请参考： :ref:`api_fluid_layers_open_files`
