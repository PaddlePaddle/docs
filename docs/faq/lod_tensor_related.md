# 关于LODTensor的问题

##### 问题：为什么 paddle2.0 以后的版本要废弃 LoDTensor ？

- 答复：在 2.0 之前的版本的 paddle 中，向用户暴露了以下的数据表示的概念：
  - [Tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/beginners_guide/basic_concept/tensor.html)： 类似于 numpy ndarray 的多维数组。
  - [Variable](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/beginners_guide/basic_concept/variable.html)：可以简单理解为，在构建静态的计算图时的数据节点。
  - [LodTensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/beginners_guide/basic_concept/lod_tensor.html)：用来表示嵌套的、每条数据长度不一的一组数据。（例：一个batch中包含了长度为3，10，7，50的四个句子）

这三类不同类型的概念的同时存在，让使用 paddle 的开发者容易感到混淆，需要构建 LoDTensor 类型的数据的情况在具体的实践中，通常也可以使用 padding/bucketing 的最佳实践来达到同样的目的，因此 paddle 2.0 版本起，我们把这些概念统一为 [Tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/01_paddle2.0_introduction/basic_concept/tensor_introduction_cn.html) 的概念。在 paddle 2.0 版本起，对于每条数据长度不一的一组数据的处理，您可以参看这篇 Tutorial： [使用注意力机制的LSTM的机器翻译](https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/nlp/seq2seq_with_attention.html)。

----------
