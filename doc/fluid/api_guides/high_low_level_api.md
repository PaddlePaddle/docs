## High/Low-level API简介

PaddlePaddle Fluid目前有2套API接口：

- Low-level（底层） API：
	
	- 灵活性强并且已经相对成熟，使用它训练的模型，能直接支持C++预测上线。
	- 提供了大量的模型作为使用示例，包括[Book](https://github.com/PaddlePaddle/book)中的全部章节，以及[models](https://github.com/PaddlePaddle/models)中的所有章节。
	- 适用人群：对深度学习有一定了解，需要自定义网络进行训练/预测/上线部署的用户。

- High-level（高层）API：
	
	- 使用简单
	- 尚未成熟，接口暂时在[paddle.fluid.contrib](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/fluid/contrib)下面。
