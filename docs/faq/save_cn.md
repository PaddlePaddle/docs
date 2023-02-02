# 模型保存常见问题


##### 问题：静态图的 save 接口与 save_inference_model 接口存储的结果有什么区别？

+ 答复：主要差别在于保存结果的应用场景：

  1. save 接口（2.0 的`paddle.static.save`或者 1.8 的`fluid.io.save`）

      该接口用于保存训练过程中的模型和参数，一般包括`*.pdmodel`，`*.pdparams`，`*.pdopt`三个文件。其中`*.pdmodel`是训练使用的完整模型 program 描述，区别于推理模型，训练模型 program 包含完整的网络，包括前向网络，反向网络和优化器，而推理模型 program 仅包含前向网络，`*.pdparams`是训练网络的参数 dict，key 为变量名，value 为 Tensor array 数值，`*.pdopt`是训练优化器的参数，结构与*.pdparams 一致。

  2. save_inference_model 接口（2.0 的`paddle.static.save_inference_model`或者 1.8 的`fluid.io.save_inference_model`）

      该接口用于保存推理模型和参数，2.0 的`paddle.static.save_inference_model`保存结果为`*.pdmodel`和`*.pdiparams`两个文件，其中`*.pdmodel`为推理使用的模型 program 描述，`*.pdiparams`为推理用的参数，这里存储格式与`*.pdparams`不同（注意两者后缀差个`i`），`*.pdiparams`为二进制 Tensor 存储格式，不含变量名。1.8 的`fluid.io.save_inference_model`默认保存结果为`__model__`文件，和以参数名为文件名的多个分散参数文件，格式与 2.0 一致。

  3. 关于更多 2.0 动态图模型保存和加载的介绍可以参考教程：[模型存储与载入](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/08_model_save_load_cn.html)

----------


##### 问题：增量训练中，如何保存模型和恢复训练？

+ 答复：在增量训练过程中，不仅需要保存模型的参数，也需要保存优化器的参数。

具体地，在 2.0 版本中需要使用 Layer 和 Optimizer 的`state_dict`和`set_state_dict`方法配合`paddle.save/load`使用。简要示例如下：

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

##### 问题：paddle.load 可以加载哪些 API 产生的结果呢？
+ 答复：

  为了更高效地使用 paddle 存储的模型参数，`paddle.load`支持从除`paddle.save`之外的其他 save 相关 API 的存储结果中载入`state_dict`，但是在不同场景中，参数`path`的形式有所不同：
    1. 从`paddle.static.save`或者`paddle.Model().save(training=True)`的保存结果载入：`path`需要是完整的文件名，例如`model.pdparams`或者`model.opt`；
    2. 从`paddle.jit.save`或者`paddle.static.save_inference_model`或者`paddle.Model().save(training=False)`的保存结果载入：`path`需要是路径前缀， 例如`model/mnist`，`paddle.load`会从`mnist.pdmodel`和`mnist.pdiparams`中解析`state_dict`的信息并返回。
    3. 从 paddle 1.x API`paddle.fluid.io.save_inference_model`或者`paddle.fluid.io.save_params/save_persistables`的保存结果载入：`path`需要是目录，例如`model`，此处 model 是一个文件夹路径。


  需要注意的是，如果从`paddle.static.save`或者`paddle.static.save_inference_model`等静态图 API 的存储结果中载入`state_dict`，动态图模式下参数的结构性变量名将无法被恢复。在将载入的`state_dict`配置到当前 Layer 中时，需要配置`Layer.set_state_dict`的参数`use_structured_name=False`。

##### 问题：paddle.save 是如何保存 state_dict，Layer 对象，Tensor 以及包含 Tensor 的嵌套 list、tuple、dict 的呢？
+ 答复：
  1. 对于``state_dict``保存方式与 paddle2.0 完全相同，我们将``Tensor``转化为``numpy.ndarray``保存。

  2. 对于其他形式的包含``Tensor``的对象（``Layer``对象，单个``Tensor``以及包含``Tensor``的嵌套``list``、``tuple``、``dict``），在动态图中，将``Tensor``转化为``tuple(Tensor.name, Tensor.numpy())``;在静态图中，将``Tensor``直接转化为``numpy.ndarray``。之所以这样做，是因为当在静态图中使用动态保存的模型时，有时需要``Tensor``的名字因此将名字保存下来，同时，在``load``时区分这个``numpy.ndarray``是由 Tenosr 转化而来还是本来就是``numpy.ndarray``；保存静态图的``Tensor``时，通常通过``Variable.get_value``得到``Tensor``再使用``paddle.save``保存``Tensor``，此时，``Variable``是有名字的，这个``Tensor``是没有名字的，因此将静态图``Tensor``直接转化为``numpy.ndarray``保存。
    > 此处动态图 Tensor 和静态图 Tensor 是不相同的，动态图 Tensor 有 name、stop_gradient 等属性；而静态图的 Tensor 是比动态图 Tensor 轻量级的，只包含 place 等基本信息，不包含名字等。

##### 问题：将 Tensor 转换为 numpy.ndarray 或者 tuple(Tensor.name, Tensor.numpy())不是惟一可译编码，为什么还要做这样的转换呢？
+ 答复：

  1. 我们希望``paddle.save``保存的模型能够不依赖 paddle 框架就能够被用户解析（pickle 格式模型），这样用户可以方便的做调试，轻松的看到保存的参数的数值。其他框架的模型与 paddle 模型做转化也会容易很多。

  2. 我们希望保存的模型尽量小，只保留了能够满足大多场景的信息（动态图保存名字和数值，静态图只保存数值），如果需要``Tensor``的其他信息（例如``stop_gradient``），可以向被保存的对象中添加这些信息，``load``之后再还原这些信息。这样的转换方式可以覆盖绝大多数场景，一些特殊场景也是可以通过一些方法解决的，如下面的问题。

##### 问题：什么情况下 save 与 load 的结果不一致呢，应该如何避免这种情况发生呢？
+ 答复：

  以下情况会造成 save 与 load 的结果不一致:
    1. 被保存的对象包含动态图``Tensor``同时包含``tuple(string, numpy.ndarray)``；
    2. 被保存的对象包含静态图``Tensor``，同时包含``numpy.ndarray``或者``tuple(string, numpy.ndarray)``；
    3. 被保存的对象只包含``numpy.ndarray``，但是包含``tuple(string, numpy.ndarray)``。

  针对这些情况我们有以下建议：
    1. 被保存的对象（包括``Layer``对象中的``ParamBase``）,避免包含形如``tuple(string, numpy.ndarray)``的对象；
    2. 如果被保存的对象包含``numpy.ndarray``，尽量在``load``时设置``return_numpy = True``。
    3. 对于``Layer``对象，只保存参数的值和名字，如果需要其他信息（例如``stop_gradient``），请将手将这些信息打包成`dict`等，一并保存。

##### 问题：paddle 2.x 如何保存模型文件？如何保存 paddle 1.x 中的 model 文件?
+ 答复：

    1. 在 paddle2.x 可使用``paddle.jit.save``接口以及``paddle.static.save_inference_model``,通过指定``path``来保存成为``path.pdmodel``和``path.pdiparams``,可对应 paddle1.x 中使用``save_inference_model``指定 dirname 和 params_filename 生成``dirname/__model__``和``dirname/params 文件``。paddle2.x 保存模型文件详情可参考:
    - [paddle.jit.save/load](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/08_model_save_load_cn.html#dongtaitumoxing-canshubaocunzairu-xunliantuili)
    - [paddle.static.save/load_inference_model](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/02_paddle2.0_develop/08_model_save_load_cn.html#jingtaitumoxing-canshubaocunzairu-tuilibushu)
    2. 如果想要在 paddle2.x 中读取 paddle 1.x 中的 model 文件，可参考:
    - [兼容载入旧格式模型](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.2rc/guides/01_paddle2.0_introduction/load_old_format_model.html#cn-guides-load-old-format-model)


##### 问题：paddle 如何单独 load 存下来所有模型变量中某一个变量，然后修改变量中的值？
+ 答复：

    1. 如果目的是修改存储变量的值，可以使用``paddle.save``保存下来所有变量，然后再使用``paddle.load``将所有变量载入后，查找目标变量进行修改，示例代码如下：

```python
import paddle

layer = paddle.nn.Linear(3, 4)
path = 'example/model.pdparams'
paddle.save(layer.state_dict(), path)
layer_param = paddle.load(path)
# 修改 fc_0.b_0 的值
layer_param["fc_0.b_0"] = 10
```

    2. 如果目的是单独访问某个变量，需要单独存储然后再单独读取，示例代码如下：

```python
import paddle

layer = paddle.nn.Linear(3, 4)
path_w = 'example/weight.tensor'
path_b = 'example/bias.tensor'
paddle.save(layer.weight, path_w)
paddle.save(layer.bias, path_b)
tensor_bias = paddle.load(path_b)
tensor_bias[0] = 10
```

更多介绍请参考以下 API 文档：

- [paddle.save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/save_cn.html#save)
- [paddle.load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/load_cn.html#load)
