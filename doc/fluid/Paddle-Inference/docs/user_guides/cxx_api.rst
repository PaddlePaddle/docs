使用C++预测
==========
为了简单方便地进行推理部署，飞桨提供了一套高度优化的C++ API推理接口。下面对各主要API使用方法进行详细介绍。    

在 `使用流程 <./tutorial.html>`_ 一节中，我们了解到Paddle Inference预测包含了以下几个方面：

- 配置推理选项
- 创建predictor
- 准备模型输入
- 模型推理
- 获取模型输出

那我们先用一个简单的程序介绍这一过程：

.. code:: c++

	std::unique_ptr<paddle::PaddlePredictor> CreatePredictor() {
		// 通过AnalysisConfig配置推理选项
		AnalysisConfig config;
		config.SetModel(“./resnet50/model”,
	                     "./resnet50/params");
		config.EnableUseGpu(100, 0);
		config.SwitchUseFeedFetchOps(false);
		config.EnableMKLDNN();
		config.EnableMemoryOptim();
		// 创建predictor
		return CreatePaddlePredictor(config);
	}
	
	void Run(paddle::PaddlePredictor *predictor,
			const std::vector<float>& input,
			const std::vector<int>& input_shape, 
			std::vector<float> *out_data) {
		// 准备模型的输入
		int input_num = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
	
		auto input_names = predictor->GetInputNames();
		auto input_t = predictor->GetInputTensor(input_names[0]);
		input_t->Reshape(input_shape);
		input_t->copy_from_cpu(input.data());
		// 模型推理
		CHECK(predictor->ZeroCopyRun());
	  
		// 获取模型的输出
		auto output_names = predictor->GetOutputNames();
		// there is only one output of Resnet50
		auto output_t = predictor->GetOutputTensor(output_names[0]);
		std::vector<int> output_shape = output_t->shape();
		int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
		out_data->resize(out_num);
		output_t->copy_to_cpu(out_data->data());
	}


以上的程序中 **CreatePredictor** 函数对推理过程进行了配置以及创建了Predictor。 **Run** 函数进行了输入数据的准备、模型推理以及输出数据的获取过程。

接下来我们依次对程序中出现的AnalysisConfig，Predictor，模型输入，模型输出做一个详细的介绍。

一：关于AnalysisConfig
------------------

AnalysisConfig管理AnalysisPredictor的推理配置，提供了模型路径设置、推理引擎运行设备选择以及多种优化推理流程的选项。配置中包括了必选配置以及可选配置。 

1. 必选配置
>>>>>>>>>>>>

**a. 设置模型和参数路径**   

从磁盘加载模型时，根据模型和参数文件存储方式不同，设置AnalysisConfig加载模型和参数的路径有两种形式：

* **non-combined形式** ：模型文件夹model_dir下存在一个模型文件和多个参数文件时，传入模型文件夹路径，模型文件名默认为__model__。 使用方式为： `config->SetModel("./model_dir")`;。
* **combined形式** ：模型文件夹model_dir下只有一个模型文件`model`和一个参数文件params时，传入模型文件和参数文件路径。 使用方式为： `config->SetModel("./model_dir/model", "./model_dir/params");`。
* 内存加载模式：如果模型是从内存加载(模型必须为combined形式)，可以使用

.. code:: c++

	std::ifstream in_m(FLAGS_dirname + "/model");
	std::ifstream in_p(FLAGS_dirname + "/params");
	std::ostringstream os_model, os_param;
	os_model << in_m.rdbuf();
	os_param << in_p.rdbuf();
	config.SetModelBuffer(os_model.str().data(), os_model.str().size(), os_param.str().data(), os_param.str().size());

Paddle Inference有两种格式的模型，分别为 **non-combined** 以及 **combined** 。这两种类型我们在 `Quick Start <../introduction/quick_start.html>`_ 一节中提到过，忘记的同学可以回顾下。

**b. 关闭Feed，Fetch op** 

config->SwitchUseFeedFetchOps(false);  // 关闭feed和fetch OP使用，使用ZeroCopy接口必须设置此项`

我们用一个小的例子来说明我们为什么要关掉它们。  
假设我们有一个模型，模型运行的序列为:
**input -> FEED_OP -> feed_out -> CONV_OP -> conv_out -> FETCH_OP -> output**                   

序列中大些字母的FEED_OP, CONV_OP, FETCH_OP 为模型中的OP， 小写字母的input，feed_out，output 为模型中的变量。                      

在ZeroCopy模式下，我们通过 	`predictor->GetInputTensor(input_names[0])` 获取的模型输入为FEED_OP的输出， 即feed_out，我们通过 `predictor->GetOutputTensor(output_names[0])` 接口获取的模型输出为FETCH_OP的输入，即conv_out，这种情况下，我们在运行期间就没有必要运行feed和fetch OP了，因此需要设置 `config->SwitchUseFeedFetchOps(false)` 来关闭feed和fetch op。


2. 可选配置
>>>>>>>>>> 

**a. 加速CPU推理**
 
.. code:: 

	// 开启MKLDNN，可加速CPU推理，要求预测库带MKLDNN功能。
	config->EnableMKLDNN();	  	  		
	// 可以设置CPU数学库线程数math_threads，可加速推理。
	// 注意：math_threads * 外部线程数 需要小于总的CPU的核心数目，否则会影响预测性能。
	config->SetCpuMathLibraryNumThreads(10); 


**b. 使用GPU推理**

.. code:: 

	// EnableUseGpu后，模型将运行在GPU上。
	// 第一个参数表示预先分配显存数目，第二个参数表示设备的ID。
	config->EnableUseGpu(100, 0); 


如果使用的预测lib带Paddle-TRT子图功能，可以打开TRT选项进行加速, 详细的请访问 `Paddle-TensorRT文档 <../optimize/paddle_trt.html>`_： 

.. code:: c++

	// 开启TensorRT推理，可提升GPU推理性能，需要使用带TensorRT的推理库
	config->EnableTensorRtEngine(1 << 30      /*workspace_size*/,   
								batch_size        /*max_batch_size*/,  
   								3                 /*min_subgraph_size*/, 
								AnalysisConfig::Precision::kFloat32 /*precision*/, 
								false             /*use_static*/, 
								false             /*use_calib_mode*/);

通过计算图分析，Paddle可以自动将计算图中部分子图融合，并调用NVIDIA的 TensorRT 来进行加速。


**c. 内存/显存优化**

.. code:: c++

	config->EnableMemoryOptim();  // 开启内存/显存复用

该配置设置后，在模型图分析阶段会对图中的变量进行依赖分类，两两互不依赖的变量会使用同一块内存/显存空间，缩减了运行时的内存/显存占用（模型较大或batch较大时效果显著）。


**d. debug开关**


.. code:: c++
	
	// 该配置设置后，会关闭模型图分析阶段的任何图优化，预测期间运行同训练前向代码一致。
	config->SwitchIrOptim(false);
	// 该配置设置后，会在模型图分析的每个阶段后保存图的拓扑信息到.dot文件中，该文件可用graphviz可视化。
	config->SwitchIrDebug();


二：关于PaddlePredictor
-----------------------
PaddlePredictor 是在模型上执行推理的预测器，根据AnalysisConfig中的配置进行创建。


.. code:: c++

	std::unique_ptr<PaddlePredictor> predictor = CreatePaddlePredictor(config);


CreatePaddlePredictor 期间首先对模型进行加载，并且将模型转换为由变量和运算节点组成的计算图。接下来将进行一系列的图优化，包括OP的横向纵向融合，删除无用节点，内存/显存优化，以及子图（Paddle-TRT）的分析，加速推理性能，提高吞吐。


三：输入输出
--------------------------

1. 准备输入
>>>>>>>>>>>>>>>>>

**a. 获取模型所有输入的tensor名字**

.. code:: c++

	std::vector<std::string> input_names = predictor->GetInputNames();

**b. 获取对应名字下的tensor**


.. code:: c++

	// 获取第0个输入
	auto input_t = predictor->GetInputTensor(input_names[0]);

**c. 将数据copy到tensor中**

.. code:: c++

	// 在copy前需要设置tensor的shape
	input_t->Reshape({batch_size, channels, height, width});
	// tensor会根据上述设置的shape从input_data中拷贝对应数目的数据到tensor中。
	input_t->copy_from_cpu<float>(input_data /*数据指针*/);

当然我们也可以用mutable_data获取tensor的数据指针:

.. code:: c++

	// 参数可为PaddlePlace::kGPU, PaddlePlace::kCPU
	float *input_d = input_t->mutable_data<float>(PaddlePlace::kGPU);


2. 获取输出
>>>>>>>>

**a. 获取模型所有输出的tensor名字**

.. code:: c++

	std::vector<std::string> out_names = predictor->GetOutputNames();

**b. 获取对应名字下的tensor**

.. code:: c++

	// 获取第0个输出
	auto output_t = predictor->GetOutputTensor(out_names[0]);

**c. 将数据copy到tensor中**

.. code:: c++

	std::vector<float> out_data;
	// 获取输出的shpae
	std::vector<int> output_shape = output_t->shape();
	int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, 	std::multiplies<int>());
	out_data->resize(out_num);
	output_t->copy_to_cpu(out_data->data());


我们可以用data接口获取tensor的数据指针：

.. code:: c++

	// 参数可为PaddlePlace::kGPU, PaddlePlace::kCPU
	int output_size;
	float *output_d = output_t->data<float>(PaddlePlace::kGPU, &output_size);

**下一步**

看到这里您是否已经对Paddle Inference的C++使用有所了解了呢？请访问 `这里 <https://github.com/PaddlePaddle/Paddle-Inference-Demo/tree/master/c%2B%2B>`_ 进行样例测试。
