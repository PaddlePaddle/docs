.. _cn_api_paddle_dataset_cifar:

cifar
-------------------------------

CIFAR数据集。

此模块将从 https://www.cs.toronto.edu/~kriz/cifar.html 下载数据集，并将训练集和测试集解析为paddle reader creator。

cifar-10数据集由10个类别的60000张32x32彩色图像组成，每个类别6000张图像。共有5万张训练图像，1万张测试图像。

cifar-100数据集与cifar-10类似，只是它有100个类，每个类包含600张图像。每个类有500张训练图像和100张测试图像。



.. py:function:: paddle.dataset.cifar.train100()

CIFAR-100训练数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[0，1]，标签范围是[0，9]。

返回： 训练数据集的reader creator。

返回类型：callable


.. py:function:: paddle.dataset.cifar.test100()

CIFAR-100测试数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[0，1]，标签范围是[0，9]。

返回： 测试数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.cifar.train10(cycle=False)

CIFAR-10训练数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[0，1]，标签范围是[0，9]。

参数：
    - **cycle** (bool) – 是否循环使用数据集

返回： 训练数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.cifar.test10(cycle=False)

CIFAR-10测试数据集的creator。

它返回一个reader creator, reader中的每个样本的图像像素范围是[0，1]，标签范围是[0，9]。

参数：
    - **cycle** (bool) – 是否循环使用数据集

返回： 测试数据集的reader creator

返回类型：callable


.. py:function:: paddle.dataset.cifar.convert(path)

将数据集转换为recordio格式。



