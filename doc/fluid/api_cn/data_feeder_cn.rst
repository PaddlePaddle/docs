###################
 fluid.data_feeder
###################



.. _cn_api_fluid_data_feeder_DataFeeder:

DataFeeder
-------------------------------

.. py:class:: paddle.fluid.data_feeder.DataFeeder(feed_list, place, program=None)



``DataFeeder`` 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 ``Executor`` 和 ``ParallelExecutor`` 中。
reader通常返回一个minibatch条目列表。在列表中每一条目都是一个样本（sample）,它是由具有一至多个特征的列表或元组组成的。


以下是简单用法：

..  code-block:: python
	
	place = fluid.CPUPlace()
	img = fluid.layers.data(name='image', shape=[1, 28, 28])
	label = fluid.layers.data(name='label', shape=[1], dtype='int64')
	feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
	result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])
	
在多GPU模型训练时，如果需要提前分别向各GPU输入数据，可以使用 ``decorate_reader`` 函数。

..  code-block:: python

	place=fluid.CUDAPlace(0)
	feeder = fluid.DataFeeder(place=place, feed_list=[data, label])
	reader = feeder.decorate_reader(
    		paddle.batch(flowers.train(), batch_size=16))



参数：  
	- **feed_list** (list) – 向模型输入的变量表或者变量表名
	- **place** (Place) – place表明是向GPU还是CPU中输入数据。如果想向GPU中输入数据, 请使用 ``fluid.CUDAPlace(i)`` (i 代表 the GPU id)；如果向CPU中输入数据, 请使用  ``fluid.CPUPlace()``
    	- **program** (Program) – 需要向其中输入数据的Program。如果为None, 会默认使用 ``default_main_program()``。 缺省值为None


弹出异常:	  ``ValueError``  – 如果一些变量不在此 Program 中


**代码示例**

..  code-block:: python

	# ...
	place = fluid.CPUPlace()
	feed_list = [
    		main_program.global_block().var(var_name) for var_name in feed_vars_name
	] # feed_vars_name 是一个由变量名组成的列表
	feeder = fluid.DataFeeder(feed_list, place)
	for data in reader():
    		outs = exe.run(program=main_program,
               		       feed=feeder.feed(data))
			       
			       
.. py:method:: feed(iterable)


根据feed_list（数据输入表）和iterable（可遍历的数据）提供的信息，将输入数据转成一种特殊的数据结构，使它们可以输入到 ``Executor`` 和 ``ParallelExecutor`` 中。

参数:	
	- **iterable** (list|tuple) – 要输入的数据

返回：  转换结果

返回类型:	dict


.. py:method:: feed_parallel(iterable, num_places=None)


该方法获取的多个minibatch，并把每个minibatch提前输入进各个设备中。

参数:	
    - **iterable** (list|tuple) – 要输入的数据
    - **num_places** (int) – 设备数目。默认为None。

返回: 转换结果

返回类型: dict

.. note::
   设备（CPU或GPU）的数目必须等于minibatch的数目



.. py:method::  decorate_reader(reader, multi_devices, num_places=None, drop_last=True)


  
将reader返回的输入数据batch转换为多个mini-batch，之后每个mini-batch都会被输入进各个设备（CPU或GPU）中。
    
参数：
        - **reader** (fun) – 该参数是一个可以生成数据的函数
        - **multi_devices** (bool) – bool型，指明是否使用多个设备
        - **num_places** (int) – 如果 ``multi_devices`` 为 ``True`` , 可以使用此参数来设置GPU数目。如果 ``num_places`` 为 ``None`` ，该函数默认使用当前训练机所有GPU设备。默认为None。
        - **drop_last** (bool) – 如果最后一个batch的大小比 ``batch_size`` 要小，则可使用该参数来指明是否选择丢弃最后一个batch数据。 默认为 ``True`` 

返回：转换结果

返回类型: dict
    
弹出异常： ValueError – 如果 ``drop_last`` 值为False并且reader返回的minibatch数目与设备数目不相等时，产生此异常


        







