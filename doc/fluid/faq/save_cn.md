# 模型保存与加载

##### 问题：增量训练中，如何保存模型和恢复训练？

+ 答复：在增量训练过程中，不仅需要保存模型的参数，也需要保存优化器的参数。

具体地，在1.8版本中需要使用Layer和Optimizer的`state_dict`和`set_dict`方法配合`fluid.save_dygraph/load_dygraph`使用。简要示例如下：

```
import paddle.fluid as fluid

with fluid.dygraph.guard():
    emb = fluid.dygraph.Embedding([10, 10])

    state_dict = emb.state_dict()
    fluid.save_dygraph(state_dict, "paddle_dy")

    adam = fluid.optimizer.Adam( learning_rate = fluid.layers.noam_decay( 100, 10000),
                                parameter_list = emb.parameters() )

    state_dict = adam.state_dict()
    fluid.save_dygraph(state_dict, "paddle_dy")

    para_state_dict, opti_state_dict = fluid.load_dygraph("paddle_dy")
    emb.set_dict(para_state_dict)
    adam.set_dict(opti_state_dict)
```

更多介绍请参考以下API文档：
- [save_dygraph](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/dygraph_cn/save_dygraph_cn.html#save-dygraph)
- [load_dygraph](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/dygraph_cn/load_dygraph_cn.html#load-dygraph)

![](https://ai-studio-static-online.cdn.bcebos.com/aba33440dd194ea397528f06bcb3574bddcf496b679b4da2832955b71cf65c76)

* 答复：报错是由于没有安装GPU版本的PaddlePaddle，CPU版本默认不包含CUDA检测功能。使用`pip install paddlepaddle-gpu -U` 即可。

-----

##### 问题：静态图的save接口与save_inference_model接口存储的结果有什么区别？

+ 答复：主要差别在于保存结果的应用场景：

  1. save接口（2.0的`paddle.static.save`或者1.8的`fluid.io.save`）

      该接口用于保存训练过程中的模型和参数，一般包括`*.pdmodel`，`*.pdparams`，`*.pdopt`三个文件。其中`*.pdmodel`是训练使用的完整模型program描述，区别于推理模型，训练模型program包含完整的网络，包括前向网络，反向网络和优化器，而推理模型program仅包含前向网络，`*.pdparams`是训练网络的参数dict，key为变量名，value为Tensor array数值，`*.pdopt`是训练优化器的参数，结构与*.pdparams一致。

  2. save_inference_model接口（2.0的`paddle.static.save_inference_model`或者1.8的`fluid.io.save_inference_model`）

      该接口用于保存推理模型和参数，2.0的`paddle.static.save_inference_model`保存结果为`*.pdmodel`和`*.pdiparams`两个文件，其中`*.pdmodel`为推理使用的模型program描述，`*.pdiparams`为推理用的参数，这里存储格式与`*.pdparams`不同（注意两者后缀差个`i`），`*.pdiparams`为二进制Tensor存储格式，不含变量名。1.8的`fluid.io.save_inference_model`默认保存结果为`__model__`文件，和以参数名为文件名的多个分散参数文件，格式与2.0一致。

  3. 关于更多2.0动态图模型保存和加载的介绍可以参考教程：[模型存储与载入](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/08_model_save_load_cn.html)

----------
