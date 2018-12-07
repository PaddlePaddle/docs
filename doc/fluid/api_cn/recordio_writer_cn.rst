#######################
 fluid.recordio_writer
#######################




.. _cn_api_fluid_recordio_writer_convert_reader_to_recordio_file:

convert_reader_to_recordio_file
-------------------------------

.. py:function::  paddle.fluid.recordio_writer.convert_reader_to_recordio_file(filename, reader_creator, feeder, compressor=Compressor.Snappy, max_num_records=1000, feed_order=None)

将 Python reader 转换为recordio文件

**代码示例：**

.. code-block:: python

	import paddle.fluid as fluid
	import paddle.dataset.mnist as mnist
	import paddle

	tmp_program = fluid.Program()
	with fluid.program_guard(tmp_program):
	    img = fluid.layers.data(name='img', shape=[784])
	    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
	feeder = fluid.DataFeeder(feed_list=[img, label], place=fluid.CPUPlace())
	# mnist.recordio 会在当前目录生成
	fluid.recordio_writer.convert_reader_to_recordio_file(
	                    filename="mnist.recordio",
	                    reader_creator=paddle.batch(mnist.train(), batch_size=32),
	                    feeder=feeder)

参数：
	- **filename** (str) - recordio文件名
	- **reader_creator** (callable) - Python reader的创造器。可参考 :ref:`api_guide_python_reader`
	- **feeder** (DataFeeder) - 数据处理实例。用于将 :code:`reader_creator` 转换为 :code:`lod_tensor`
	- **compressor** – 必须在 :code:`fluid.core.RecordIOWriter.Compressor.Snappy` 或 :code:` fluid.core.RecordIOWriter.Compressor.NoCompress` 中， 默认情况下使用 :code:`Snappy`
	- **max_num_records** (int) – 一个 chuck 中 records 的最大数量。每个 records 都是 reader 函数的返回值
	- **feed_order** (list) - reader 返回的变量名的顺序

返回： 保存的 record 的数目

返回类型： int

英文版API文档：:ref:`api_fluid_recordio_writer_convert_reader_to_recordio_file`







.. _cn_api_fluid_recordio_writer_convert_reader_to_recordio_files:

convert_reader_to_recordio_files
-------------------------------

.. py:function:: paddle.fluid.recordio_writer.convert_reader_to_recordio_files(filename, batch_per_file, reader_creator, feeder, compressor=Compressor.Snappy, max_num_records=1000, feed_order=None)

该函数可以将一个python驱动的reader（数据读取器）转变为多个recodio文件。

该API实现的功能和 ``convert_reader_to_recordio_file`` 基本相同，只不过本函数会生成多个recordio文件。
每个文件最多存储 ``batch_per_file`` 条记录。

请参照 :ref:`cn_api_fluid_recordio_writer_convert_reader_to_recordio_file` 获取更详细的介绍。

英文版API文档: :ref:`api_fluid_recordio_writer_convert_reader_to_recordio_files` 







