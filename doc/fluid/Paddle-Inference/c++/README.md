# C++ 预测样例

**如果您看到这个目录，我们会假设您已经对Paddle Inference有了一定的了解。**

**如果您刚刚接触Paddle Inference不久，建议您[访问这里](https://paddle-inference.readthedocs.io/en/latest/#)对Paddle Inference做一个初步的认识。**

这个目录包含了图像中使用的分类，检测，以及NLP中Ernie/Bert模型测试样例，同时也包含了Paddle-TRT，多线程等测试样例。

为了能够顺利运行样例，请您在环境中准备Paddle Inference C++预编译库。

**一：获取编译库：**

- [官网下载](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html)。
- 自行编译获取。

**二：预编译lib目录结构介绍：**

进入预编译库，目录结构为：

```
├── CMakeCache.txt
├── paddle
├── third_party
└── version.txt
```

其中`paddle`目录包含了预编译库的头文件以及lib文件。  
`third_party`包含了第三方依赖库的头文件以及lib文件。

`version.txt`包含了lib的相关描述信息，包括：

	```
	GIT COMMIT ID: 06897f7c4ee41295e6e9a0af2a68800a27804f6c
	WITH_MKL: ON         # 是否带MKL
	WITH_MKLDNN: OFF     # 是否带MKLDNN
	WITH_GPU: ON         # 是否支持GPU
	CUDA version: 10.1   # CUDA的版本
	CUDNN version: v7。  # CUDNN版本
	WITH_TENSORRT: ON    # 是否带TRT
	```


有了预编译库后我们开始进入各个目录进行样例测试吧～
