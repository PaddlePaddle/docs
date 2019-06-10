## Introduction to High/Low-level API

Currently PaddlePaddle Fluid has 2 branches of API interfaces:

- Low-level API:

	- It is highly flexible and relatively mature. The model trained by it can directly support C++ inference deployment and release.
	- There are a large number of models as examples, including all chapters in [book](https://github.com/PaddlePaddle/book), and [models](https://github.com/PaddlePaddle/models).
	- Recommended for users who have a certain understanding of deep learning and need to customize a network for training/inference/online deployment.

- High-level API:

	- Simple to use
    - Still under development. the interface is temporarily in [paddle.fluid.contrib](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/contrib).