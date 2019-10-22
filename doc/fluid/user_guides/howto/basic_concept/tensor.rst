.. _cn_user_guide_tensor:

=========
Tensor
=========

飞桨（PaddlePaddle，以下简称Paddle）和其他框架一样，使用Tensor来表示数据。

在神经网络中传递的数据都是Tensor,Tensor可以简单理解成一个多维数组，一般而言可以有任意多的维度。不同的Tensor可以具有自己的数据类型和形状，同一Tensor中每个元素的数据类型是一样的，Tensor的形状就是Tensor的维度。

下图直观地表示1～6维的Tensor：
<p align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/FluidDoc/develop/doc/fluid/beginners_guide/image/tensor.jpg" width="400">
</p>


**Paddle 高级特性**  

:ref:`Lod_Tensor <cn_user_guide_lod_tensor>` 

对于一些任务中batch内样本大小不一致的问题，Paddle提供了两种解决方案：1: padding， 将大小不一致的样本padding到同样的大小，这是一种常用且推荐的使用方式； 2：:ref:`Lod_Tensor <cn_user_guide_lod_tensor>` ，记录每一个样本的大小，减少无用的计算量，Lod 牺牲灵活性来提升性能。

如果一个batch内的样本无法通过分桶、排序等方式使得大小接近， 建议使用lod tensor。
