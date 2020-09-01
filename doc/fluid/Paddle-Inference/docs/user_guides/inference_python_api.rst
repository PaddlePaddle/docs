使用Python预测
===============

Paddle Inference提供了高度优化的Python 和C++ API预测接口，本篇文档主要介绍Python API，使用C++ API进行预测的文档可以参考可以参考 `这里 <./cxx_api.html>`_ 。

下面是详细的使用说明。

使用Python预测API预测包含以下几个主要步骤：

- 配置推理选项
- 创建Predictor
- 准备模型输入
- 模型推理
- 获取模型输出

我们先从一个简单程序入手，介绍这一流程：

.. code:: python

	def create_predictor():
		# 通过AnalysisConfig配置推理选项
		config = AnalysisConfig("./resnet50/model", "./resnet50/params")
		config.switch_use_feed_fetch_ops(False)
		config.enable_use_gpu(100, 0)
		config.enable_mkldnn()
		config.enable_memory_optim()
		predictor = create_paddle_predictor(config)
		return predictor

	def run(predictor, data):
		# 准备模型输入
		input_names = predictor.get_input_names()
		for i,  name in enumerate(input_names):
			input_tensor = predictor.get_input_tensor(name)
			input_tensor.reshape(data[i].shape)
			input_tensor.copy_from_cpu(data[i].copy())

		# 执行模型推理
		predictor.zero_copy_run()

		results = []
		# 获取模型输出
		output_names = predictor.get_output_names()
		for i, name in enumerate(output_names):
			output_tensor = predictor.get_output_tensor(name)
			output_data = output_tensor.copy_to_cpu()
			results.append(output_data)

		return results


以上的程序中 **create_predictor** 函数对推理过程进行了配置以及创建了Predictor。 **run** 函数进行了输入数据的准备、模型推理以及输出数据的获取过程。

在接下来的部分中，我们会依次对程序中出现的AnalysisConfig，Predictor，模型输入，模型输出进行详细的介绍。

一、推理配置管理器AnalysisConfig
----------------------------
AnalysisConfig管理AnalysisPredictor的推理配置，提供了模型路径设置、推理引擎运行设备选择以及多种优化推理流程的选项。配置中包括了必选配置以及可选配置。

1. 必选配置
>>>>>>>>>>>>

**a.设置模型和参数路径**

* **Non-combined形式**：模型文件夹 model_dir 下存在一个模型文件和多个参数文件时，传入模型文件夹路径，模型文件名默认为__model__。 使用方式为： `config.set_model("./model_dir")`

* Combined形式：模型文件夹 model_dir 下只有一个模型文件 model 和一个参数文件params时，传入模型文件和参数文件路径。使用方式为： `config.set_model("./model_dir/model", "./model_dir/params")`

* 内存加载模式：如果模型是从内存加载，可以使用:

	.. code:: python
		
		import os
		model_buffer = open('./resnet50/model','rb')
		params_buffer = open('./resnet50/params','rb')
		model_size = os.fstat(model_buffer.fileno()).st_size
		params_size = os.fstat(params_buffer.fileno()).st_size
		config.set_model_buffer(model_buffer.read(), model_size, params_buffer.read(), params_size)


关于 non-combined 以及 combined 模型介绍，请参照 `这里 <../introduction/quick_start.html>`_。

**b. 关闭feed与fetch OP**

config.switch_use_feed_fetch_ops(False)  # 关闭feed和fetch OP

2. 可选配置
>>>>>>>>>
 
**a. 加速CPU推理**
 
.. code:: python

	# 开启MKLDNN，可加速CPU推理，要求预测库带MKLDNN功能。
	config.enable_mkldnn()	  	  		
	# 可以设置CPU数学库线程数math_threads，可加速推理。
	# 注意：math_threads * 外部线程数 需要小于总的CPU的核心数目，否则会影响预测性能。
	config.set_cpu_math_library_num_threads(10) 


**b. 使用GPU推理**

.. code:: python

	# enable_use_gpu后，模型将运行在GPU上。
	# 第一个参数表示预先分配显存数目，第二个参数表示设备的ID。
	config.enable_use_gpu(100, 0) 

如果使用的预测lib带Paddle-TRT子图功能，可以打开TRT选项进行加速： 

.. code:: python


	# 开启TensorRT推理，可提升GPU推理性能，需要使用带TensorRT的推理库
	config.enable_tensorrt_engine(1 << 30,    # workspace_size
			batch_size,    # max_batch_size
			3,    # min_subgraph_size
			AnalysisConfig.Precision.Float32,    # precision
			False,    # use_static
			False,    # use_calib_mode
			)

通过计算图分析，Paddle可以自动将计算图中部分子图融合，并调用NVIDIA的 TensorRT 来进行加速。
使用Paddle-TensorRT 预测的完整方法可以参考 `这里 <../optimize/paddle_trt.html>`_。


**c. 内存/显存优化**

.. code:: python

	config.enable_memory_optim()  # 开启内存/显存复用

该配置设置后，在模型图分析阶段会对图中的变量进行依赖分类，两两互不依赖的变量会使用同一块内存/显存空间，缩减了运行时的内存/显存占用（模型较大或batch较大时效果显著）。


**d. debug开关**


.. code:: python

	# 该配置设置后，会关闭模型图分析阶段的任何图优化，预测期间运行同训练前向代码一致。
	config.switch_ir_optim(False)


.. code:: python

	# 该配置设置后，会在模型图分析的每个阶段后保存图的拓扑信息到.dot文件中，该文件可用graphviz可视化。
	config.switch_ir_debug(True)

二、预测器PaddlePredictor
----------------------

PaddlePredictor 是在模型上执行推理的预测器，根据AnalysisConfig中的配置进行创建。

.. code:: python
	
	predictor = create_paddle_predictor(config)


create_paddle_predictor 期间首先对模型进行加载，并且将模型转换为由变量和运算节点组成的计算图。接下来将进行一系列的图优化，包括OP的横向纵向融合，删除无用节点，内存/显存优化，以及子图（Paddle-TRT）的分析，加速推理性能，提高吞吐。


三：输入/输出
---------------

1.准备输入
>>>>>>>>>>>>

**a. 获取模型所有输入的Tensor名字**

.. code:: python

	input_names = predictor.get_input_names()

**b. 获取对应名字下的Tensor**

.. code:: python

	# 获取第0个输入
	input_tensor = predictor.get_input_tensor(input_names[0])

**c. 将输入数据copy到Tensor中**

.. code:: python

	# 在copy前需要设置Tensor的shape
	input_tensor.reshape((batch_size, channels, height, width))
	# Tensor会根据上述设置的shape从input_data中拷贝对应数目的数据。input_data为numpy数组。
	input_tensor.copy_from_cpu(input_data)


2.获取输出
>>>>>>>>>

**a. 获取模型所有输出的Tensor名字**

.. code::python

	output_names = predictor.get_output_names()

**b. 获取对应名字下的Tensor**

.. code:: python
	
	# 获取第0个输出
	output_tensor = predictor.get_output_tensor(ouput_names[0])

**c. 将数据copy到Tensor中**

.. code:: python
	
	# output_data为numpy数组
	output_data = output_tensor.copy_to_cpu()


**下一步**

看到这里您是否已经对 Paddle Inference 的 Python API 使用有所了解了呢？请访问 `这里 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/python>`_ 进行样例测试。
