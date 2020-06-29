.. _cn_api_paddle_dataset_mnist:

mnist
-------------------------------

MNIST数据集。

此模块将从 http://yann.lecun.com/exdb/mnist/ 下载数据集，并将训练集和测试集解析为paddle reader creator。



.. py:function:: paddle.dataset.mnist.train()

MNIST训练数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[-1，1]，标签范围是[0，9]。

返回： 训练数据的reader creator

返回类型：callable



.. py:function:: paddle.dataset.mnist.test()

MNIST测试数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[-1，1]，标签范围是[0，9]。

返回： 测试数据集的reader creator

返回类型：callable



.. py:function:: paddle.dataset.mnist.convert(path)

将数据集转换为recordio格式。



