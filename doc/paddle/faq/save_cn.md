# 模型保存与加载


##### 问题：静态图的save接口与save_inference_model接口存储的结果有什么区别？

+ 答复：主要差别在于保存结果的应用场景：

  1. save接口（2.0的`paddle.static.save`或者1.8的`fluid.io.save`）

      该接口用于保存训练过程中的模型和参数，一般包括`*.pdmodel`，`*.pdparams`，`*.pdopt`三个文件。其中`*.pdmodel`是训练使用的完整模型program描述，区别于推理模型，训练模型program包含完整的网络，包括前向网络，反向网络和优化器，而推理模型program仅包含前向网络，`*.pdparams`是训练网络的参数dict，key为变量名，value为Tensor array数值，`*.pdopt`是训练优化器的参数，结构与*.pdparams一致。

  2. save_inference_model接口（2.0的`paddle.static.save_inference_model`或者1.8的`fluid.io.save_inference_model`）

      该接口用于保存推理模型和参数，2.0的`paddle.static.save_inference_model`保存结果为`*.pdmodel`和`*.pdiparams`两个文件，其中`*.pdmodel`为推理使用的模型program描述，`*.pdiparams`为推理用的参数，这里存储格式与`*.pdparams`不同（注意两者后缀差个`i`），`*.pdiparams`为二进制Tensor存储格式，不含变量名。1.8的`fluid.io.save_inference_model`默认保存结果为`__model__`文件，和以参数名为文件名的多个分散参数文件，格式与2.0一致。

  3. 关于更多2.0动态图模型保存和加载的介绍可以参考教程：[模型存储与载入](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/08_model_save_load_cn.html)

----------


##### 问题：增量训练中，如何保存模型和恢复训练？

+ 答复：在增量训练过程中，不仅需要保存模型的参数，也需要保存优化器的参数。

具体地，在2.0版本中需要使用Layer和Optimizer的`state_dict`和`set_state_dict`方法配合`paddle.save/load`使用。简要示例如下：

```
import paddle

emb = paddle.nn.Embedding(10, 10)
layer_state_dict = emb.state_dict()
paddle.save(layer_state_dict, "emb.pdparams")

scheduler = paddle.optimizer.lr.NoamDecay(
    d_model=0.01, warmup_steps=100, verbose=True)
adam = paddle.optimizer.Adam(
    learning_rate=scheduler,
    parameters=emb.parameters())
opt_state_dict = adam.state_dict()
paddle.save(opt_state_dict, "adam.pdopt")

load_layer_state_dict = paddle.load("emb.pdparams")
load_opt_state_dict = paddle.load("adam.pdopt")

emb.set_state_dict(para_state_dict)
adam.set_state_dict(opti_state_dict)
```

更多介绍请参考以下API文档：

- [paddle.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/framework/io/save_cn.html)
- [paddle.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/framework/io/load_cn.html)
