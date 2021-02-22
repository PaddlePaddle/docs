# 数据及其加载常见问题


##### 问题：如何在训练过程中高效读取数量很大的数据集？

+ 答复：当训练时使用的数据集数据量较大或者预处理逻辑复杂时，如果串行地进行数据读取，数据读取往往会成为训练效率的瓶颈。这种情况下通常需要利用多线程或者多进程的方法异步地进行数据载入，从而提高数据读取和整体训练效率。

paddle1.8中推荐使用两个异步数据加载的API：

1. DataLoader.from_generator，有限的异步加载

该API提供了单线程和单进程的异步加载支持。但由于线程和进程数目不可配置，所以异步加速能力是有限的，适用于数据读取负载适中的场景。

具体使用方法及示例请参考API文档：[fluid.io.DataLoader.from_generator](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/io_cn/DataLoader_cn.html#from-generator-feed-list-none-capacity-none-use-double-buffer-true-iterable-true-return-list-false-use-multiprocess-false-drop-last-true)。

2. DataLoader，灵活的异步加载

该API提供了多进程的异步加载支持，也是paddle后续主推的数据读取方式。用户可通过配置num_workers指定异步加载数据的进程数目从而满足不同规模数据集的读取需求。

具体使用方法及示例请参考API文档：[fluid.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/io_cn/DataLoader_cn.html#paddle.fluid.io.DataLoader)

----------

##### 问题：使用多卡进行并行训练时，如何配置DataLoader进行异步数据读取？

+ 答复：paddle1.8中多卡训练时设置异步读取和单卡场景并无太大差别，动态图模式下，由于目前仅支持多进程多卡，每个进程将仅使用一个设备，比如一张GPU卡，这种情况下，与单卡训练无异，只需要确保每个进程使用的是正确的卡即可。

具体示例请参考飞桨API [fluid.io.DataLoader.from_generator](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/io_cn/DataLoader_cn.html#from-generator-feed-list-none-capacity-none-use-double-buffer-true-iterable-true-return-list-false-use-multiprocess-false-drop-last-true) 和 [fluid.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/io_cn/DataLoader_cn.html#paddle.fluid.io.DataLoader) 中的示例。

----------


##### 问题：有拓展Tensor维度的Op吗？

+ 答复：请参考API [paddle.fluid.layers.unsqueeze](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/unsqueeze_cn.html#paddle.fluid.layers.unsqueeze)。

----------


##### 问题：如何给图片添加一个通道数，并进行训练？

+ 答复：如果是在进入paddle计算流程之前，数据仍然是numpy.array的形式，使用numpy接口`numpy.expand_dims`为图片数据增加维度后，再通过`numpy.reshape`进行操作即可，具体使用方法可查阅numpy的官方文档。

如果是希望在模型训练或预测流程中完成通道的操作，可以使用paddle对应的API [paddle.fluid.layers.unsqueeze](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/unsqueeze_cn.html#paddle.fluid.layers.unsqueeze) 和 [paddle.fluid.layers.reshape](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/layers_cn/reshape_cn.html#reshape)。

----------


##### 问题：如何从numpy.array生成一个具有shape和dtype的Tensor?

+ 答复：在动态图模式下，可以参考如下示例：

```
import paddle.fluid as fluid

with fluid.dygraph.guard(fluid.CPUPlace()):
    x = np.ones([2, 2], np.float32)
    y = fluid.dygraph.to_variable(x)
```

具体请参考API [paddle.fluid.dygraph.to_variable](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.8/api_cn/dygraph_cn/to_variable_cn.html#to-variable)

----------

##### 问题：如何初始化一个随机数的Tensor？

+ 答复：使用`numpy.random`生成随机的numpy.array，再使用`numpy.random`参考上一个问题中的示例创建随机数Tensor即可。
