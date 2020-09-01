# Paddle Inference Demos



Paddle Inference为飞桨核心框架推理引擎。Paddle Inference功能特性丰富，性能优异，针对服务器端应用场景进行了深度的适配优化，做到高吞吐、低时延，保证了飞桨模型在服务器端即训即用，快速部署。


为了能让广大用户快速的使用Paddle Inference进行部署应用，我们在此Repo中提供了C++、Python的使用样例。


**在这个repo中我们会假设您已经对Paddle Inference有了一定的了解。**

**如果您刚刚接触Paddle Inference不久，建议您[访问这里](https://paddle-inference.readthedocs.io/en/latest/#)对Paddle Inference做一个初步的认识。**


## 测试样例

1） 在python目录中，我们通过真实输入的方式罗列了一系列的测试样例，其中包括图像的分类，分割，检测，以及NLP的Ernie/Bert等Python使用样例，同时也包含Paddle-TRT， 多线程的使用样例。

2） 在c++目录中，我们通过单测方式展现了一系列的测试样例，其中包括图像的分类，分割，检测，以及NLP的Ernie/Bert等C++使用样例，同时也包含Paddle-TRT， 多线程的使用样例。
