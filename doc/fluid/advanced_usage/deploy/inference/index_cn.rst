############
服务器端部署
############

推理（Inference）指的是在设备上运行训练好的模型，依据输入数据来进行预测。Paddle Fluid提供了预测库及其C++和Python的API来支持模型的部署上线。
使用Paddle Fluid预测主要包含以下几个步骤：

    1. 加载由Paddle Fluid训练的模型和参数文件；
    2. 准备输入数据。即将待预测的数据（如图片）转换成Paddle Fluid模型接受的格式，并将其设定为预测引擎的输入；
    3. 运行预测引擎，获得模型的输出；
    4. 根据业务需求解析输出结果，获得需要的信息。

以上步骤使用的API会在后续部分进行详细介绍。

.. toctree::
   :titlesonly:

   build_and_install_lib_cn.rst
   native_infer.md
   python_infer_cn.md
   paddle_tensorrt_infer.md
   paddle_gpu_benchmark.md
   windows_cpp_inference.md
