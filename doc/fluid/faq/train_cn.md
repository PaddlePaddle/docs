# 网络搭建及训练



## 模型介绍

##### Q: sigmoid中num_classes意义？

+ 问题描述

sigmoid二分类，`sigmoid_cross_entropy_with_logits`，其中num_classes的意义是什么？

+ 问题解答

`sigmoid_cross_entropy_with_logits`里面的num_classes是指有多个分类标签，而且这些标签之间相互独立，这样对每个分类都会有一个预测概率。举个例子，假如我们要做一个视频动作分类，有如下几个标签（吃饭，聊天，走路，打球），那么num_classes = 4。一个视频可以同时有多个ground truth标签是1，比如这里可能是(1, 1, 0, 0)，也就是一边吃饭一边聊天的场景。而一个可能的预测概率是(0.8, 0.9, 0.1, 0.3)，那么计算损失函数的时候，要对每个分类标签分别计算`sigmoid cross entropy`。

##### Q：proto信息解释文档？

+ 问题描述

PaddlePaddle的proto信息相关的解释文档？

+ 问题解答

proto信息可以打印出来，然后参考Github [framework.prot](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto)文件进行解释。


## 模型调用

##### Q: 如何不训练某层的权重？

+ 问题解答

ParamAttr里配置`learning_rate=0`。

##### Q: 根据输出结束模型运行？

+ 问题描述

PaddlePaddle可以像tf一样根据输出，只执行模型的一部分么？

+ 问题解答

目前的executor会执行整个program。建议做法是，在定义program的时候提前clone出一个子program，如当前`inference_program = fluid.default_main_program().clone(for_test=True)`的做法。

##### Q: 遍历每一个时间布？

+ 问题描述

在RNN模型中如何遍历序列数据里每一个时间步？

+ 问题解答

对于LodTensor数据，分步处理每一个时间步数据的需求，大部分情况可以使用`DynamicRNN`，[参考示例](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/tests/book/test_machine_translation.py#L69) ，其中`rnn.step_input`即是每一个时间步的数据，`rnn.memory`是需要更新的rnn的hidden state，另外还有个`rnn.static_input`是rnn外部的数据在DynamicRNN内的表示（如encoder的output），可以利用这三种数据完成所需操作。

rank_table记录了每个sequence的长度，DynamicRNN中调用了lod_tensor_to_array在产生array时按照rank_table做了特殊处理（sequence从长到短排序后从前到后进行slice），每个时间步数据的batch size可能会缩小（短的sequence结束时），这是Fluid DynamicRNN的一些特殊之处。
对于非LoDTensor数据，可以使用StaticRNN，用法与上面类似，参考[语言模型示例]( https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleNLP/language_model/lstm/lm_model.py#L261)。

##### Q: NumPy读取出fc层W矩阵？

+ 问题描述

PaddlePaddle中训练好的fc层参数，有没有直接用numpy 读取出fc层的W矩阵的示例代码呢？

+ 问题解答

weight名字在构建网络的时候可以通过param_attr指定，然后用`fluid.global_scope().find_var("weight_name").get_tensor()`。

##### Q: `stop_gradient=True`影响范围？

+ 问题描述

请问fluid里面如果某一层使用`stop_gradient=True`，那么是不是这一层之前的层都会自动 `stop_gradient=True`?

+ 问题解答

是的，梯度不回传了。

##### Q: 如何获取accuracy？

+ 问题描述

根据models里面的[ctr例子](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleRec/ctr/infer.py)改写了一个脚本，主要在train的同时增加test过程，方便选择模型轮次，整体训练测试过程跑通，不过无法获取accuracy？

+ 问题解答

AUC中带有LOD 信息，需要设置`return_numpy=False `来获得返回值。

##### Q: 图片小数量大数据集处理方法

+ 问题描述

对于图片小但数量很大的数据集有什么好的处理方法？

+ 问题解答

`multiprocess_reader`可以解决该问题。参考[Github示例](https://github.com/PaddlePaddle/Paddle/issues/16592)。

##### Q: Libnccl.so报错如何解决？

+ 报错信息

```
Failed to find dynamic library: libnccl.so ( libnccl.so: cannot open shared object file: No such file or directory ）
```

+ 问题解答

按照以下步骤做检查：

1、先确定是否安装libnccl.so

2、确定环境变量是否配置正确

```
export LD_LIBRARY_PATH=`pwd`/nccl_2.1.4-1+cuda8.0_x86_64/lib:$LD_LIBRARY_PATH
```

3、确定是否要配置软链

```
cd /usr/lib/x86_64-linux-gnu
ln -s libnccl.so.2 libnccl.so
```


## 模型保存

##### Q: 保存模型API选择

+ 问题描述

请说明一下如下两个接口的适用场景，现在保存模型都不知用哪个：`save_inference_model`、`save_params`？

+ 问题解答

`save_inference_model`主要是用于预测的，该API除了会保存预测时所需的模型参数，还会保存预测使用的模型结构。而`save_params`会保存一个program中的所有参数，但是不保存该program对应的模型结构。参考[模型保存与加载](http://paddlepaddle.org/documentation/docs/zh/1.4/api_guides/low_level/model_save_reader.html)

##### Q: 保存模型报错

+ 问题描述

CTR模型保存模型时报错

+ 代码文件：[network_conf.py](https://github.com/PaddlePaddle/models/blob/develop/fluid/PaddleRec/ctr/network_conf.py)只修改了最后一行：

```
accuracy = fluid.layers.accuracy(input=predict, label=words[-1])
auc_var, batch_auc_var, auc_states = \
    fluid.layers.auc(input=predict, label=words[-1], num_thresholds=2 ** 12, slide_steps=20)
return accuracy, avg_cost, auc_var, batch_auc_var, py_reader
```

+ 问题解答

保存模型时需指定program 才能正确保存。请使用`executor = Executor(place)`, 你的train_program, 以及给`layers.data`指定的名称作为`save_inference_model` 的输入。


## 参数相关

##### Q: 手动输入参数并改变？

+ 问题描述

PaddlePaddle的全连接层，可不可以手动输入参数比如weights和bias并禁止优化器比如optimizer.SGD在模型训练的时候改变它？

+ 问题解答

可以通过ParamAttr设置参数的属性，`fluid.ParamAttr( initializer=fluid.initializer.Normal(0.0, 0.02), learning_rate=2.0)`，其中learning_rate设置为0，就不会修改。手动输入参数也可以实现，但是会比较麻烦。

##### Q: `fluid.unique_name.guard()`影响范围

+ 问题描述

batch norm 里面的两个参数：moving_mean_name、moving_variance_name应该是两个var，但是他们却没有受到 `with fluid.unique_name.guard()` 的影响，导致名字重复？

+ 问题解答

用户指定的name的优先级高于unique_name生成器，所以名字不会被改变。

##### Q: 2fc层共享参数？

+ 问题描述

怎么配置让两个fc层共享参数？

+ 问题解答

只要指定param_attr相同名字即可，是`param_attr = fluid.ParamAttr(name='fc_share')`，然后把param_attr传到fc里去。


## LoD-Tensor数据结构相关

##### Q: 拓展tensor纬度

+ 问题描述

PaddlePaddle有拓展tensor维度的op吗？

+ 问题解答

请参[unsqueeze op](http://paddlepaddle.org/documentation/docs/zh/1.3/api/layers.html#unsqueeze)，例如[1,2]拓展为[1，2，1]

##### Q: 多维变长tensor?

+ 问题描述

PaddlePaddle是否支持两维以上的变长tensor，如shape[-1, -1, 128]？

+ 问题解答

配置网络的时候可以将shape写成[-1,任意正数,128]，然后输入的时候shape可以为[任意正数,任意正数,128]。维度只是个占位，运行网络的时候的实际维度是从输入数据推导出来的。两个"任意整数" 在输入和配置的时候可以不相等。配置网络的时候第一维度必须为-1。

##### Q: vector -> LodTensor

+ 问题描述

C++ 如何把std::vector转换成LodTensor的方法?

+ 问题解答

如下示例

```cpp
std::vector<int64_t> ids{1918, 117, 55, 97, 1352, 4272, 1656, 903};
framework::LoDTensor words;
auto size = static_cast<int>(ids.size());
framework::LoD lod{{0, ids.size()}};
DDim dims{size, 1};
words.Resize(dims);
words.set_lod(lod);
auto *pdata = words.mutable_data<int64_t>();
size_t n = words.numel() * sizeof(int64_t);
memcpy(pdata, ids.data(), n);
```

##### Q: 报错holder should not be null

+ 错误信息

```
C++ Callstacks:
holder should not be null
Tensor not initialized yet when Tensor::type() is called. at [/paddle/paddle/fluid/framework/tensor.h:145]
PaddlePaddle Call Stacks:
```

+ 问题解答

错误提示是某个tensor为空。建议运行时加上环境变量GLOG_vmodule=operator=4 , GLOG_logtostderr=1看看运行到哪个op，哪个tensor为空。


## pyreader

##### Q: 加载模型时pyreader使用

+ 问题描述

调用`save_inference_model`后，`load_inference_model`加载预测模型的时候用py_reader读取，`feeded_var_names`为空也无法通过feed输入了。py_reader此刻应该如何声明？

+ 问题解答

目前`load_inference_model`加载进行的模型还不支持py_reader输入。

##### Q: 变量取名

+ 问题描述

使用py_reader读取数据的时候，怎么给读取的变量指定名字呢？

+ 问题解答

参考[create_py_reader_by_data](http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#create-py-reader-by-data)
