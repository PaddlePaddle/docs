# Release Note

## 重要更新

飞桨框架2.1.0 版本有如下重要更新：

- 环境适配： 增加了对Python 3.9、CUDA 11.2的支持；提供了对[ROCm平台](https://rocmdocs.amd.com/en/latest/)的支持（experimental）；提供了对[昇腾AI处理器](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/ascend-910)的支持（experimental）；增加了可在[百度昆仑芯片](https://cloud.baidu.com/product/kunlun.html)上运行的模型数量；详情请见：[开始使用](https://www.paddlepaddle.org.cn/install/quick)。

- 分布式训练：分布式支持多维混合并行功能，含数据并行，模型并行，流水线并行，分组参数切片策略，及与AMP混合精度策略，ReCompute策略的任意组合。

- 框架功能：完成了多项功能增强和性能优化，特别的，新增了以下重要功能：
    - 提供了在框架外部自定义算子的新方案，简化了自定义算子写法与训练推理部署流程，详情请见：[自定义外部算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/07_new_op/new_custom_op_cn.html)。
    - 新增可降低显存占用与提升性能的inplace操作，包括View策略，与12个inplace API。
    - 新增支持混合精度训练的高层API；新增通过`paddle.hub`来查看、共享、加载模型。
    - 自动混合精度训练优化： 优化了混合精度训练中slice、where、range等多个op的计算性能，提升了在MaskRCNN、ERNIE等模型上的加速效果。
    - oneDNN下BF16训练：新增支持了AMP(AutoMixedPrecision) pure_BF16模式; 新增支持了BF16类型的SGD和initializers初始值设定并减小了内存；新增支持了大部分word2vec BF16训练需要的前向和反向op。
- 模型库及开发套件：飞桨的官方模型库和套件的最新更新请参见：[Paddle projects notes along with PaddlePaddle2.1](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-projects-notes-along-with-PaddlePaddle2.1)。
  
## 不兼容升级
- 飞桨框架2.1放弃了对python2和python3.5的支持，建议您升级python到3.8版本来使用飞桨。飞桨框架2.1不再提供支持CUDA9的预编译包，建议您升级CUDA版本来使用飞桨。
- 对API可见性的优化，会导致无法使用`from deeply_nested_namespace import *`的方式导入被认为是实现细节的位于最底层的命名空间中的私有API。建议您通过查看飞桨官网的[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)说明来使用飞桨。具体的，以下行为在飞桨框架2.1版本中不再被允许。

```python
# will import nothing from the deeply nested namespaces
from paddle.nn.layer.loss import *
from paddle.nn.layer.conv import *
```

- `Tensor.grad`不兼容升级，返回值的类型由`numpy`变为`Tensor`。([#32142](https://github.com/PaddlePaddle/Paddle/pull/32142))

| 2.0                                                          | 2.1                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| import paddle<br /> x = paddle.to_tensor(5., stop_gradient=False)<br /> y = paddle.pow(x, 4.0)<br /> y.backward()<br /> type(x.grad)<br /> < class 'numpy.ndarray' >| import paddle<br /> x = paddle.to_tensor(5., stop_gradient=False)<br /> y = paddle.pow(x, 4.0)<br /> y.backward()<br /> type(x.grad)<br />< class 'paddle.Tensor' > |


- `paddle.jit.TraceLayer.save_inference_model` 接口不兼容升级。将原先的第一个参数dirname改为path，名字表意更通用并且与paddle.save和load等接口统一，表示由用户指定保存模型路径的前缀。([#31989](https://github.com/PaddlePaddle/Paddle/pull/31989))
  
  | 2.0                                                          | 2.1                                                          |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | import os<br />import paddle<br />from paddle.vision.models import resnet18<br /><br />model = resnet18()<br />x = paddle.rand([1, 3, 224, 224])<br />_, static_layer = paddle.jit.TracedLayer.trace(model, input=[x])<br />save_path = './save_infer_model' <br />static_layer.save_inference_model(**dirname**=save_path) <br /><br />print(os.path.isdir(save_path))<br />print(len(os.listdir(save_path)))<br /><br /> True<br />205| import os<br />import paddle<br />from paddle.vision.models import resnet18<br /><br />model = resnet18()<br />x = paddle.rand([1, 3, 224, 224])<br />_, static_layer = paddle.jit.TracedLayer.trace(model, input=[x])<br />save_path = './save_infer_model' <br />static_layer.save_inference_model(**path**=save_path) <br /><br />print(os.path.isdir(save_path))<br />print([name for name in os.listdir('./') if name.startswith(save_path)])<br /><br /> False <br />`[save_infer_model.pdiparams]`|


- `paddle.io.DataLoader`当`Dataset`只包含一个字段时，`DataLoader`返回格式不兼容升级。当用户自定义数据集只包含一个字段并通过如`return image`或`yield image`返回数据时，2.0版本返回的数据格式是`[image_tensor]`，而2.1版本返回的数据格式为`image_tensor`，保持输入输出数据结构的一致性。

  | 2.0                                                          | 2.1                                                          |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | import numpy as np<br />import paddle<br />from paddle.io import DataLoader, Dataset<br /><br />class RandomDataset(Dataset):<br />def \_\_getitem\_\_(self, idx):<br />return np.random.random((2, 3)).astype('float32')<br /><br />def \_\_len\_\_(self): <br />return 10<br /><br />dataset = RandomDataset()<br />loader = DataLoader(dataset, batch_size=1)<br /> data = next(loader())<br /># data: [Tensor(shape=(1, 2, 3), dtype=float32)]|import numpy as np<br />import paddle<br />from paddle.io import DataLoader, Dataset<br /><br />class RandomDataset(Dataset):<br />def \_\_getitem\_\_(self, idx):<br />return np.random.random((2, 3)).astype('float32')<br /><br />def \_\_len\_\_(self): <br />return 10<br /><br />dataset = RandomDataset()<br />loader = DataLoader(dataset, batch_size=1)<br /> data = next(loader())<br /># data: Tensor(shape=(1, 2, 3), dtype=float32)|


## 训练框架

### 功能优化（含分布式）

#### 基础API

- 新增`paddle.dtype` 以及 `paddle.float32` 等数据类型，作为 paddle 内的数据类型。 ([#32012](https://github.com/PaddlePaddle/Paddle/pull/32012)) 
- 新增`paddle.nn.functional.glu`。 ([#32096](https://github.com/PaddlePaddle/Paddle/pull/32096)) 
- 新增`paddle.nn.utils.spectral_norm`。[#32633](https://github.com/PaddlePaddle/Paddle/pull/32633)
- 新增`paddle.Tensor.register_hook` API，用于在动态图场景中为前向Tensor对应的梯度Tensor注册hook函数。([#31775](https://github.com/PaddlePaddle/Paddle/pull/31775))
- 新增`Tensor.__array__`函数，支持`numpy.array(Tensor)`和`numpy.asarray(Tensor)`将`paddle.Tensor`类型转换成`numpy.ndarray`类型 。([#32300](https://github.com/PaddlePaddle/Paddle/pull/32300))
- 新增Tensor API：``Tensor.item(*args)``，可将Tensor中指定位置的元素转化为Python的scalar值并返回。([#32634](https://github.com/PaddlePaddle/Paddle/pull/32634))
- 新增`paddle.nn.LayerList`对负数索引的支持。([#31750](https://github.com/PaddlePaddle/Paddle/pull/31750))
- 新增12个动态图inplace API：`clip_`、`scale_`、`add_`、`subtract_`、`ceil_`、`floor_`、`exp_`、`reciprocal_`、`round_`、`sqrt_`、`rsqrt_`、`flatten_`。这些inplace API不能通过`paddle.api_`的形式调用，应该使用`Tensor.api_`来调用。([#32699](https://github.com/PaddlePaddle/Paddle/pull/32699))
- 新增`paddle.autograd.backward` API,  用于自定义起始梯度。([#31540](https://github.com/PaddlePaddle/Paddle/pull/31540))
- 新增`paddle.nn.LayerDict` 类。([#31951](https://github.com/PaddlePaddle/Paddle/pull/31951))
- 新增`layer.to` API。([#32040](https://github.com/PaddlePaddle/Paddle/pull/32040))
- 新增`paddle.autograd.PyLayer`API，用于支持动态图在Python端自定义反向计算。([#32130](	https://github.com/PaddlePaddle/Paddle/pull/32130))
- 新增支持`paddle.optimizer`在动态图中指定非参数的Tensor作为parameters进行优化。[#32362](https://github.com/PaddlePaddle/Paddle/pull/32362))
- 在`paddle.static.nn`添加了若干 `sequence*` 系列功能，在 `paddle.nn.functional` 添加了`sequence_mask`。 ([#32089](https://github.com/PaddlePaddle/Paddle/pull/32089))
- 在`paddle.nn.CTCLoss`中添加`norm_by_times`参数。([#32490](https://github.com/PaddlePaddle/Paddle/pull/32490))
- `paddle.fill_constant` 支持 `uint8_t`。([#31911](https://github.com/PaddlePaddle/Paddle/pull/31911))
- `paddle.clip`支持`int32`和`int64`。([#32373](https://github.com/PaddlePaddle/Paddle/pull/32373))
- 支持`paddle.nn.functional.interpolate` 在 Nearest neighbor 模式下，输入数据类型为int。([#32270](https://github.com/PaddlePaddle/Paddle/pull/32270))
- API中所有支持传入list或tuple的参数，全部升级为支持传入list和tuple。([#32344](https://github.com/PaddlePaddle/Paddle/pull/32344),  [#32528](https://github.com/PaddlePaddle/Paddle/pull/32528) [#32360](https://github.com/PaddlePaddle/Paddle/pull/32360))
- 优化`softmax`算子性能。([#31821](https://github.com/PaddlePaddle/Paddle/pull/31821))
- 优化`paddle.norm`文档说明，澄清`paddle.norm`与`numpy.linalg.norm`API 存在功能差异。([#32530](https://github.com/PaddlePaddle/Paddle/pull/32530)) 
- 优化Tensor 的数据类型（`datatype`）的打印形式，例如，`float32`类型的Tensor的`dtype`从`VarType.FP32`变为 `paddle.float32`。([#30682](https://github.com/PaddlePaddle/Paddle/pull/30682))
- oneDNN功能优化：
 - 升级 oneDNN 至 v2.2.1。([#31067](https://github.com/PaddlePaddle/Paddle/pull/31067) [#31473])(https://github.com/PaddlePaddle/Paddle/pull/31473), [#30295](https://github.com/PaddlePaddle/Paddle/pull/30295) [32227](https://github.com/PaddlePaddle/Paddle/pull/32227))
 - 增加了更加准确的，基于数据类型的 oneDNN kernel 选择策略。([#29840](https://github.com/PaddlePaddle/Paddle/pull/29840))
 - 融合oneDNN `layer_norm`子图为完整的单个`layer_norm` op。([#32162](https://github.com/PaddlePaddle/Paddle/pull/32162), [#30891](https://github.com/PaddlePaddle/Paddle/pull/30891), [#30962](https://github.com/PaddlePaddle/Paddle/pull/30962))
 - 减少oneDNN `elementwise_mul`创建中不必要的内存分配。([#30203](https://github.com/PaddlePaddle/Paddle/pull/30203))
 - 改进了缓存每个线程使用的内存消耗。([#30358](https://github.com/PaddlePaddle/Paddle/pull/30358))
 - 增加了LSTM oneDNN fp32 and int8 kernel支持。([#30719](https://github.com/PaddlePaddle/Paddle/pull/30719) [#31894](https://github.com/PaddlePaddle/Paddle/pull/31894))
 - 增加了 OneDNN hardswish 支持。([#30211](https://github.com/PaddlePaddle/Paddle/pull/30211))
 - 增加了 `bilinear_interp_v2` 和 `nearest_interp_v2` 的oneDNN支持。([#32312](https://github.com/PaddlePaddle/Paddle/pull/32312))
- 升级 Xbyak 数学库 至 v5.81。([#30809](https://github.com/PaddlePaddle/Paddle/pull/30809))
- 修复`paddle.io.DataLoader`支持数据集包含list，dict和string等嵌套的复杂数据格式，修复迭代中途程序退出偶现的报错、资源未释放等问题。([#31481](https://github.com/PaddlePaddle/Paddle/pull/31481))
- 修复 paddle 中修改 logging 库的 root logger 导致的问题。([#32706](https://github.com/PaddlePaddle/Paddle/pull/32706))
- 修复`L1Decay`动态图模式下`backward`报错的问题。([32718](https://github.com/PaddlePaddle/Paddle/pull/32718))
- 修复`paddle.nn.functional.cross_entropy`中设置`ignore_index`和`reduction='mean'`下出Nan的问题。([#32545](https://github.com/PaddlePaddle/Paddle/pull/32545))
- 修复bool tensor和float tensor相加输出的类型为bool的问题。([#32272](https://github.com/PaddlePaddle/Paddle/pull/32272))
- 修复比较类API在broadcast的计算错误。([#32470](https://github.com/PaddlePaddle/Paddle/pull/32470))
- 修复加减乘除在右侧输入是大shape下的broadcast下梯度计算错误。([#30818](https://github.com/PaddlePaddle/Paddle/pull/30818))
- 修复segment mean OP在处理大shape tensor输入时，计算结果不正确的问题。([#32610](https://github.com/PaddlePaddle/Paddle/pull/32610))
- 修复优化器变量的数据类型与模型参数的数据类型不一致的问题。([#29917](https://github.com/PaddlePaddle/Paddle/pull/29917)) 
- 修复 `paddle.io.DataLoader`预处理中包含paddle的操作时，`num worker>0`时报错。([#31177](https://github.com/PaddlePaddle/Paddle/pull/31177))
- 修复打印空tensor时的报错。([#32501](https://github.com/PaddlePaddle/Paddle/pull/32501))
- 调整静态图参数初始化顺序，调整后与动态图保持一致，以便于相同模型设置相同随机种子在动态图和静态图中初始化得到相同参数。([#32177](https://github.com/PaddlePaddle/Paddle/pull/32177))
- 修复`paddle.to_tensor` 不支持接受`dtype=Tensor.dtype`的bug。([#31931](https://github.com/PaddlePaddle/Paddle/pull/31931)) 
- 修复`paddle.dist` 在2个输入相等时，梯度为nan的问题。([#32448](https://github.com/PaddlePaddle/Paddle/pull/32448))
- `paddle.nn.functional.temporal_shift` API增加`data_format`属性，支持设置为NCHW或者NHWC。([#31642](https://github.com/PaddlePaddle/Paddle/pull/31642))
- 修复`adaptive_avg_pool2d`在输入数据类型为float16时计算结果不正确的问题。([#31887](https://github.com/PaddlePaddle/Paddle/pull/31887))
- `paddle.nn.Layer.sublayers` 和`paddle.nn.Layer.named_sublayers`：将原本`paddle.nn.Layer.sublayers`的`include_sublayers = True`参数修改为`include_self = False`， 从而修复从前`include_sublayers = False`时返回空的问题。现在不填写任何参数时默认行为和之前一致，即返回不包含自己的所有递归子层，当`include_self = True`时同字面意思，返回包含自己在内的所有递归子层。而`paddle.nn.Layer.named_sublayers`中`include_sublayers`的参数则直接删除了 其他行为不变。([#31824](https://github.com/PaddlePaddle/Paddle/pull/31824) )

| 2.0                                                          | 2.1                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| from paddle.vision.models import resnet18<br/>model = resnet18()<br/><br/>print(len(model.sublayers(include_sublayers=True)))<br/>print(len(model.sublayers(include_sublayers=False)))<br/><br/>67<br/>0<br/> | from paddle.vision.models import resnet18<br/>model = resnet18()<br/><br/>print(len(model.sublayers(include_self=True)))<br/>print(len(model.sublayers(include_self=False)))<br/><br/>68<br/>67<br/> |
 

#### 高层API
- 新增`paddle.hub`功能，提供`help`、`list`和`load`函数用于查看和加载第三方模型，支持加载远程和本地repository。([#31873](https://github.com/PaddlePaddle/Paddle/pull/31873))
- 支持混合精度训练，提供O0, O1, O2三种模式，分别对应FP32训练、自动混合精度训练、纯FP16训练。目前纯FP16训练仅支持静态图。([#31417](	https://github.com/PaddlePaddle/Paddle/pull/31417))
- 支持`paddle.Tensor`类型的图像变换，包括`normalize, to_grayscale, vflip, hflip, crop, center_crop, pad, rotate, resize`等算子 。([#32705](https://github.com/PaddlePaddle/Paddle/pull/32705))


#### 动态图转静态图
修复了动态图转静态图的bug：

- 静态图`arange、range` API返回的shape与动态图不一致。
- `paddle.to_tensor`在动转静中支持输入为`int，float，bool`基础类型。
- for循环中支持解析dict推导式语法。([#32159](https://github.com/PaddlePaddle/Paddle/pull/32159))
- 修复部分场景下嵌套控制流语句中存在变量未声明报错的问题。([#32153](https://github.com/PaddlePaddle/Paddle/pull/32153)) 
- 修复了`expand` op缺少float16类型的bug。([#32238](https://github.com/PaddlePaddle/Paddle/pull/32238))
- 修复了`expand_v2、tile、expand、expand_as、expand_as_v2、meshgrid`等6个OP反向梯度求解，当shape维度为6时，返回梯度信息为None的bug。([#32004](https://github.com/PaddlePaddle/Paddle/pull/32004))
- 修复了`paddle.jit.TraceLayer.save_inference_model`接口中因未同时保存网络结构和参数导致与`paddle.static.load_inference_model`搭配使用不一致的问题。([#31989](https://github.com/PaddlePaddle/Paddle/pull/31989) )

#### 混合精度训练

- 动态图混合精度接口 auto_cast 中自动将不支持fp16 kernel的op保持为fp32计算。([#32543](https://github.com/PaddlePaddle/Paddle/pull/32543)) 
- 修复静态图混合精度训练中因不支持FP16计算的Op列表(`unsupported_fp16_list`)统计不完整导致的意外报错问题，当前不支持FP16计算的Op列表可根据运行时环境自动生成。([#32102](https://github.com/PaddlePaddle/Paddle/pull/32102))
- 优化`update_loss_scaling` for循环起多个相同cuda kernel问题，融合为一个cuda kernel。([#32554](https://github.com/PaddlePaddle/Paddle/pull/32554))
- 优化`slice`多维情况下性能较慢问题。([#32266](https://github.com/PaddlePaddle/Paddle/pull/32266))
- 优化`elementwise_add_grad`输入输出相同时的冗余拷贝问题。([#32051](https://github.com/PaddlePaddle/Paddle/pull/32051))
- 优化`check_finite_and_unscale` for循环起多个相同cuda kernel问题，融合为一个cuda kernel。([#31954](https://github.com/PaddlePaddle/Paddle/pull/31954))
- 优化`range`参数冗余拷贝问题。([#30811](https://github.com/PaddlePaddle/Paddle/pull/30811))
- 优化`top_k_v2`在`input_width <= 1024`时性能较慢问题。([#30403](https://github.com/PaddlePaddle/Paddle/pull/30403))
- 移植`where_index` CPU计算流程到GPU上完成。([#30601](https://github.com/PaddlePaddle/Paddle/pull/30601))

#### BF16训练
- 增加了初级 BF16 AMP 集成, 通过在前向网络中添加`cast op`来修改图使一些 operator 使用 BF16 kernel 。([#31093](https://github.com/PaddlePaddle/Paddle/pull/31093))
- 增加了 BF16 `pure_mode`模式, 在此模式下，默认开启使用 BF16 数据类型的模型参数，BF16的operator，对于optimizer的BF16 decorator。([#32281](https://github.com/PaddlePaddle/Paddle/pull/32281), [#32681](https://github.com/PaddlePaddle/Paddle/pull/32681))
- 增加了对于CPU flags的检查以确认是否支持oneDNN BF16性能提升。([#30551](https://github.com/PaddlePaddle/Paddle/pull/30551))
- 对BF16支持进行过程统一。([#31034](https://github.com/PaddlePaddle/Paddle/pull/31034))
- 增加了对于constant initilizer的BF16数据类型的支持。([#31935](https://github.com/PaddlePaddle/Paddle/pull/31935))
- 增加了BF16 uniform initializer支持。([#32468](https://github.com/PaddlePaddle/Paddle/pull/32468))
- 增加了将startup_program initializer转化为BF16的机制。([#32720](https://github.com/PaddlePaddle/Paddle/pull/32720))
- 增加了 sgd operator 的 BF16 数据类型支持。([#32162](https://github.com/PaddlePaddle/Paddle/pull/32162))
- 增加了lookup_table op BF16 数据类型的支持。([#31558](https://github.com/PaddlePaddle/Paddle/pull/31558))
- 增加了 sum kernel 和 SelectedRows 的 BF16的支持。([#32755](https://github.com/PaddlePaddle/Paddle/pull/32755),  [#32631](https://github.com/PaddlePaddle/Paddle/pull/32631))
- 增加了conv_transpose的BF16数据类型支持。([#30877](https://github.com/PaddlePaddle/Paddle/pull/30877))
- 增加了elementwise_add grad BF16数据类型的支持。([#30925](https://github.com/PaddlePaddle/Paddle/pull/30925))	
- 增加了reshape grad BF16 数据类型的支持。([#31035](https://github.com/PaddlePaddle/Paddle/pull/31035))
- 增加了elementwise_add grad op 对于 broadcasting 的支持(FP32/BF16)。([#31385](https://github.com/PaddlePaddle/Paddle/pull/31385))
- 增加了elementwise_mul grad op 对于fp32/bf16数据类型的支持。([#31647](https://github.com/PaddlePaddle/Paddle/pull/31647))
- 增加了 LSTM BF16 支持，并修复GRU BF16的一些问题。([#31234](https://github.com/PaddlePaddle/Paddle/pull/31234))
- 增加了 oneDNN reduce_op fp32 和 bf16支持。([#31816](https://github.com/PaddlePaddle/Paddle/pull/31816))
- 增加了oneDNN reduce_op grad 对于 fp32 和 bf16 的支持。([#32280](https://github.com/PaddlePaddle/Paddle/pull/32280)  [#32592](https://github.com/PaddlePaddle/Paddle/pull/32592))

#### 分布式训练优化
 
 - 加入图检索引擎，支持万亿边规模的分布式图神经网络训练([#31226](https://github.com/PaddlePaddle/Paddle/pull/31226))。
 - 加入基于索引的数据采样类，支持图、TDM/OTM树等模型的采样([#31696](https://github.com/PaddlePaddle/Paddle/pull/31696))。
 - 新增`paddle.distributed.send, paddle.distributed.recv`，完善分布式通信API。([#32504](https://github.com/PaddlePaddle/Paddle/pull/32504))
 - 新增`paddle.distributed.new_group` 和 `paddle.distributed.wait`。([#31682](https://github.com/PaddlePaddle/Paddle/pull/31682))
 - 动态图分布式初始化支持`sync_parameters_buffer`，解决动态图buffer未全局初始化的问题。([#31625](https://github.com/PaddlePaddle/Paddle/pull/31625))
 - [混合并行] Fleet静态图支持数据并行/Sharding/流水线并行/模型并行 4级混合并行。([#32486](https://github.com/PaddlePaddle/Paddle/pull/32486)，[#32485](https://github.com/PaddlePaddle/Paddle/pull/32485)，[#31996](https://github.com/PaddlePaddle/Paddle/pull/31996)，[#31939](https://github.com/PaddlePaddle/Paddle/pull/31939)，[#31796](https://github.com/PaddlePaddle/Paddle/pull/31796))
 - 流水线并行支持1F1B调度方式，优化显存占用量，理论上显存占用量为常量。（[#31786](https://github.com/PaddlePaddle/Paddle/pull/31786)）
 - [混合并行] 优化Sharding 策略：Gradient Merge支持、减少参数通信量等，提升训练速度。([#31884](https://github.com/PaddlePaddle/Paddle/pull/31884))
 - [混合并行] Sharding策略中添加optimize offload支持，降低训练显存占用。([#32134](https://github.com/PaddlePaddle/Paddle/pull/32134)) 
 - [混合并行] 持久化广播通信ID的socket服务，减少混合并行端口冲突问题。([#31589](https://github.com/PaddlePaddle/Paddle/pull/31589)) 
 - [参数服务器] 优化日志输出和LOG打印，去除无效日志。
 - [参数服务器] 优化稀疏参数存储结构，维度较小(低于64)的情况下内存有较大降幅 。
 - [参数服务器] 修复在分布式预测时，准入策略生效的BUG。
 - HeterPs支持多机。

##### 动态图混合并行
动态图分布式支持混合并行功能，支持数据并行，模型并行以及流水线并行三种并行方式的任意组合。同时支持混合并行基础上添加AMP混合精度策略，ReCompute策略。

- Fleet支持动态图混合并行，支持数据并行（DataParallel）/模型并行（ModelParallel）/流水线并行（PipelineParallel）三种并行的互相组合。([#32248](https://github.com/PaddlePaddle/Paddle/pull/32248))
- 动态图分布式DataParallel添加`find_unused_parameters`参数，用于支持控制流组网。([#31625](https://github.com/PaddlePaddle/Paddle/pull/31625))
- Fleet添加`VocabParallelEmbedding`，`ColumnParallelLinear`，`RowParallelLinear` API用于模型并行组网。添加`model_parallel_random_seed` / `get_rng_state_tracker`用于ModelParallel的随机性控制。([#32248](https://github.com/PaddlePaddle/Paddle/pull/32248))
- Fleet添加`distributed_scaler` 接口，用于混合并行AMP策略下的loss scaler。([#32354](https://github.com/PaddlePaddle/Paddle/pull/32354))
- Fleet添加`PipelineLyaer`用于流水线并行组网切图，添加`LayerDesc`用于动态图Layer描述以减少显存初始化。([#32449](https://github.com/PaddlePaddle/Paddle/pull/32449))
- 动态图新增 Recompute 策略。([#32516](https://github.com/PaddlePaddle/Paddle/pull/32516))


#### 自定义OP

- 新增支持Mac平台上使用自定义OP功能。([#31976](https://github.com/PaddlePaddle/Paddle/pull/31976))。
- Mac平台下支持C++/v11头文件目录的自动搜索功能，兼容本地可能存在多版本clang的情况。
- 新增支持Op前反向函数Attribute参数以及inferShape, InferDtype函数输入参数使用const &类型。([#31588](https://github.com/PaddlePaddle/Paddle/pull/31588))
- 新增支持在自定义Op实现时使用三种框架内部数据类型`paddle::complex64, paddle::complex128, paddle::float16`。([#31602](https://github.com/PaddlePaddle/Paddle/pull/31602), [#31657](https://github.com/PaddlePaddle/Paddle/pull/31657), [#31669](https://github.com/PaddlePaddle/Paddle/pull/31669), [#31725](https://github.com/PaddlePaddle/Paddle/pull/31725))
- 新增支持在自定义Op中使用`std::vector<paddle::Tensor>`类型参数作为前反向函数的输入。([#31535](https://github.com/PaddlePaddle/Paddle/pull/31535))
- 新增支持InferShape函数使用Attribute参数作为输入。([#31713](https://github.com/PaddlePaddle/Paddle/pull/31713))
- 优化自动生成的Python API在动态图下的调用栈，提升执行效率。([#32209](https://github.com/PaddlePaddle/Paddle/pull/32209))
- 降低Windows上检查编译器cl.exe时的报错条件，增强Windows环境自检的鲁棒性。([#32769](https://github.com/PaddlePaddle/Paddle/pull/32769))
- 修复Windows上安装多个CUDA环境时编译器选择时的bug。([#31694](https://github.com/PaddlePaddle/Paddle/pull/31694))
- 修复Windows安装中文版本VS时出现的Python编码问题的bug。([#31493](https://github.com/PaddlePaddle/Paddle/pull/31493)) 
- 移除对单独动态库文件的依赖，仅链接框架核心动态库文件。([#32404](https://github.com/PaddlePaddle/Paddle/pull/32404)、[#32769](https://github.com/PaddlePaddle/Paddle/pull/32769))
- 移除之前的旧自定义OP方案，并对whl包中多余的库文件与头文件进行了清理，降低了whl包大小约11M。([#31813](https://github.com/PaddlePaddle/Paddle/pull/31813)), ([#32463](https://github.com/PaddlePaddle/Paddle/pull/32463))


#### 模型保存与载入
- `paddle.save, paddle.load`支持Tensor的保存加载。([#31756](https://github.com/PaddlePaddle/Paddle/pull/31756))
- `paddle.save, paddle.load`支持`list[Tensor]、dict[Tensor]、tuple[Tensor]`以及`list、tuple、dict`嵌套的包含Tensor的结构的保存加载。([#32446](https://github.com/PaddlePaddle/Paddle/pull/32446))
- `paddle.save, paddle.load`支持Layer的保存加载。([#32446](https://github.com/PaddlePaddle/Paddle/pull/32446))
- `paddle.save, paddle.load`支持Program的保存加载。([#32336](	https://github.com/PaddlePaddle/Paddle/pull/32336))
- `paddle.save, paddle.load`支持C++二进制格式单个Tensor的保存加载。([#32211](https://github.com/PaddlePaddle/Paddle/pull/32211))
- `paddle.jit.save, paddle.jit.load`支持无参数的Fucntion的保存加载。([#32430](	https://github.com/PaddlePaddle/Paddle/pull/32430))

### 性能优化（含分布式）
- 优化重点算子，提升多个模型单GPU训练性能，Deeplabv3+单卡FP32和AMP性能分别提升11%、72%，TSM单卡AMP性能提升44.5%，HRNet单卡FP32、AMP分别提升46%、51%。
- 增加 `index_sample` CUDA实现。([#30380](https://github.com/PaddlePaddle/Paddle/pull/30380))
- 实现`relu, leaky_relu`算子的CUDA Kernel，代替原Eigen实现，正反向共提升5% ～ 20%。([#31869](https://github.com/PaddlePaddle/Paddle/pull/31869), [#31841](https://github.com/PaddlePaddle/Paddle/pull/31841))
- `temporal_shift` 性能提升20%～40%。([#31642](https://github.com/PaddlePaddle/Paddle/pull/31642))
- 优化`depthwise_conv2d`，NHWC format下性能提升30%～50%。([#31667](https://github.com/PaddlePaddle/Paddle/pull/31677))
- 优化`interp_bilinear_grad`算子NCHW性能，提升19%~303%。([#30950](https://github.com/PaddlePaddle/Paddle/pull/30950))
- 优化`adaptive_avg_pool2d`算子NCHW、output_size = 1情况下的性能，提升80%~90% 。([#31197](https://github.com/PaddlePaddle/Paddle/pull/31197)) 
- conv op当dtype为float16时，forward和backward支持开启`exhaustive_search`。([#30959](https://github.com/PaddlePaddle/Paddle/pull/30959))
- `momentum`的`weight_decay`参数设置为float类型时，实现`momentum`和`L2Decay`的融合。([#30881](https://github.com/PaddlePaddle/Paddle/pull/30881))
- 实现`log_softmax`算子`axis`为最后一维、维度<=1024时的CUDA Kernel，相比原Eigen实现，正反向算子性能提升4.55x ~ 26.45x。([#31630](https://github.com/PaddlePaddle/Paddle/pull/31630), [#32180](https://github.com/PaddlePaddle/Paddle/pull/32180))

## 推理部署

### 模型量化

- 新增支持将FP32模型保存为FP16模型。([#32112](https://github.com/PaddlePaddle/Paddle/pull/32112))
- 重构动态图量化训练中统计输出量化信息模块，支持多Block和多分支的模型，增强通用性。([#31680](https://github.com/PaddlePaddle/Paddle/pull/31680) [#31710](https://github.com/PaddlePaddle/Paddle/pull/31710) [#31784](https://github.com/PaddlePaddle/Paddle/pull/31784) [#31861](https://github.com/PaddlePaddle/Paddle/pull/31861))
- 动态图量化训练功能支持跳过量化OP，并且和预测端形成打通。([#31704](https://github.com/PaddlePaddle/Paddle/pull/31704))


### Paddle Inference

#### 功能升级

 - 发布C API (experimental)， 功能与C++ API基本对齐。([#32225](https://github.com/PaddlePaddle/Paddle/pull/32225))
 -  重构Tensor 底层代码，与旧有 ZeroCopyTensor 数据结构解耦。此升级不涉及用户 API 改动，对用户透明。([#31402](https://github.com/PaddlePaddle/Paddle/pull/31402))
 - 预测框架python接口接入训练自定义算子。用户在训练过程中加载自定义算子后，即可像框架原生算子那样，通过 PaddlePredictor 直接执行包含此自定义算子的预测模型部署。([#32533](https://github.com/PaddlePaddle/Paddle/pull/32533)) 
 - 支持从内存加载模型时TensorRT序列化和反序列化功能。([#31342](https://github.com/PaddlePaddle/Paddle/pull/31342))

#### 性能优化
- 支持ERNIE量化模型在NV GPU上混合精度推理，其中MatMul以Int8精度计算，其他部分以FP16精度计算。相比纯FP16推理，在T4上batch size=40时，标准ERNIE模型在XNLI数据集上推理性能由1898 seq/s提升至2310 seq/s，提升17.8%。([#32232](https://github.com/PaddlePaddle/Paddle/pull/32232))

#### 易用性优化
- 用户开启TensorRT变长输入，输入shape超出限定范围时增加报错信息。([#32155](https://github.com/PaddlePaddle/Paddle/pull/32155))
- 增加运行时TensorRT版本检查，若运行和编译时TensorRT大版本号不一致会以warning提示。([#32443](https://github.com/PaddlePaddle/Paddle/pull/32443))
- 增加TensorRT VERBOSE级别log开关，用户可通过`export GLOG_v=3`开启TensorRT VERBOSE日志，打印更多调试信息。([#32459](https://github.com/PaddlePaddle/Paddle/pull/32459))


#### BugFix
- 修复预测结束后可能出现非指定使用显卡显存不足的错误。([#32655](https://github.com/PaddlePaddle/Paddle/pull/32655))
- 修复动态图下原生推理非正规值引起的CPU推理性能问题。([#32350](https://github.com/PaddlePaddle/Paddle/pull/32350)) 
- 修复在使用PaddleSlim量化模型开启TensorRT推理时，若从内存读入模型，仍会要求设置校准表路径的问题。([#32676](https://github.com/PaddlePaddle/Paddle/pull/32676))
- 升级TensorRT量化校准表接口，修复在DLA上不支持TensorRT离线量化的问题。([#31060](https://github.com/PaddlePaddle/Paddle/pull/31060))
- 修复当使用变长方式进行ERNIE/BERT模型推理时（EnableTensorRtOSS），不支持裁剪Attention的header数量的问题。([#31497](https://github.com/PaddlePaddle/Paddle/pull/31497))
- 修复2.0之后训练的BERT模型QK输入顺序不稳定带来的结果偶现diff问题。([#32659](https://github.com/PaddlePaddle/Paddle/pull/32659))
- 修复ERNIE模型开启TensorRT varlen加速时因输入变量名顺序错误导致报错或结果错误问题。([#32482](https://github.com/PaddlePaddle/Paddle/pull/32482))
- 修复TensorRT的plugin ElementwisePluginDynamic序列化失败的问题。([#31587](https://github.com/PaddlePaddle/Paddle/pull/31587))
- 修复TensorRT动态shape下FC layer维度补1带来的后续OP维度报错的问题。([#32458](https://github.com/PaddlePaddle/Paddle/pull/32458), [#31803](https://github.com/PaddlePaddle/Paddle/pull/31803))
- 修复FC使用Padding时`repeated_fc_relu_fuse_pass.cc`错误的问题。([#32648](https://github.com/PaddlePaddle/Paddle/pull/32648/files))
- 修复conv2d_transpose op使用TensorRT推理时结果错误的问题。([#32593](https://github.com/PaddlePaddle/Paddle/pull/32593))
- 修复NAN的错误比较导致的 OCR INT8 模型 oneDNN 预测报错的问题。([#32227](https://github.com/PaddlePaddle/Paddle/pull/32227))
- 修复部署多个模型在多executor上多线程进行oneDNN 预测时出现数据争用的问题。([#32499](https://github.com/PaddlePaddle/Paddle/pull/32499),  [#32136](https://github.com/PaddlePaddle/Paddle/pull/32136) [#32664](https://github.com/PaddlePaddle/Paddle/pull/32664))


## 环境适配

### 编译安装

- 新增支持CUDA11.2编译，支持3070/3080/3090显卡架构的编译。([#31529](https://github.com/PaddlePaddle/Paddle/pull/31529))
- 新增支持Windows Visual Studio 2017编译，并将发版、CI/CE、编译文档等各项配套设施，由VS2015全面升级至VS2017。([#311652](https://github.com/PaddlePaddle/Paddle/pull/31652))
- 新增对cuda11.2镜像的支持。([#32531](https://github.com/PaddlePaddle/Paddle/pull/32531))
- cuda10.1镜像支持gcc 5.4。([#32531](https://github.com/PaddlePaddle/Paddle/pull/32531))
- 镜像中新增对python 3.9的支持。([#32385](https://github.com/PaddlePaddle/Paddle/pull/32385))
- 修复`run_check`接口的bug，并在`run_check`接口里新增了对动态图的检查：现在`run_check`检测paddle安装的逻辑里，首先检测用户机器上是否有GPU，没有则报warning，未考虑安装cpu包的用户。([#32428](https://github.com/PaddlePaddle/Paddle/pull/32428))
- 修复Windows系统上缺乏 symlink 方法的问题。([#31006](https://github.com/PaddlePaddle/Paddle/pull/31006))

### 新硬件训练支持

- 新增支持海光芯片：飞桨基于 ROCM 4.0.1 版本可以在海光CPU与DCU上进行模型训练与推理。已经验证支持图像分类、目标检测、图像分割、自然语言处理、推荐系统、视频分类与语音合成共计7个分类的36个模型。
- 新增支持昇腾芯片：支持在昇腾NPU上进行单机多卡训练。
- 昆仑硬件训练支持
	- 昆仑XPU支持动态图分布式训练。([#30455](https://github.com/PaddlePaddle/Paddle/pull/30455),  [#30671](https://github.com/PaddlePaddle/Paddle/pull/30671))
	- 昆仑XPU支持fleet分布式训练。([#30858](https://github.com/PaddlePaddle/Paddle/pull/30858))
	- 昆仑XPU支持spawn启动多卡训练，优化XPU动态图多卡性能。([#31130](https://github.com/PaddlePaddle/Paddle/pull/31130))
	- 昆仑XPU静态图多卡支持fuse allreduce及gradient merge优化。([#31104](https://github.com/PaddlePaddle/Paddle/pull/31104)) 
	- 支持昆仑XPU暴露all_reduce/reduce集合通信API。([#32303](https://github.com/PaddlePaddle/Paddle/pull/32302))
	- 修复昆仑XPU动态图多卡随机hang住的bug。([#32662](https://github.com/PaddlePaddle/Paddle/pull/32662))


## Thanks to our Contributors

This release contains contributions from:

123malin, Adam Osewski, alncat,  arlesniak, AshburnLee, Aurelius84, Bai Yifan, Baibaifan, Bin Lu, cc, ceci3, chajchaj, chalsliu, channings, Chen Long, Chen Weihang, chen zhiyu, Chengmo, chentianyu03, cnn, CtfGo, cucuzg, danleifeng, denglin-github, Double\_V, fangshuixun007, Feiyu Chan, fluffyrita, FlyingQianMM, FNRE, furnace, GaoWei8, GeminiCarrie, gongweibao, Gradie, GT-Zhang, Guanghua Yu, Guo Sheng, guofei, hong, houj04, huangjun12, huangxu96, Huihuang Zheng, hutuxian, iducn, Jacek Czaja, Jack Zhou, jakpiase, JamesLim, Jiabin Yang, jiangcheng, Jiaqi Liu, Jiawei Wang, joanna.wozna.intel, joejiong, JZ-LIANG, Kaipeng Deng, Kqnonrime, kuizhiqing, Lei.C, Leo Chen, lidanqing, LielinJiang, lijianshe02, lilong12, limingshu, littletomatodonkey, liu zhengxi, LiuChiachi, liuyuhui, liym27, LoveAn, LutaoChu, minghaoBD, mls1999725, niuliling123, Ouyang Chao, pangyoki, parap1uie-s, Pei Yang, procr, Qi Li, qingqing01, QingshuChen, Ren Wei (任卫), ronnywang, ruri, seemingwang, Shang Zhizhou, shanliang1992, ShenLiang, Shibo Tao, Steffy-zxf, syyxsxx, taixiurong, tangwei12, Tao Luo, Thomas Young, Thunderbrook, tianshuo78520a, TTerror, wangchaochaohu, wangguanzhong, wanghuancoder, wangna11BD, WangXi, wangxinxin08, wawltor, Wei Shengyu, weihaoji, WeiXin, wenbin, Wenyu, whs, Wilber, winter-wang, Wojciech Uss, wuhuanzhou, wuyefeilin, XGZhang, XiangGao, XiaoguangHu, xiaoting, xiegegege, xiemoyuan, xingfeng01, Yang Zhang, yaoxuefeng, yiak, yingshengBD, yinhaofeng, Yiqun Liu, ykkk2333, yongqiangma, Yuang Liu, yukavio, YUNSHEN XIE, Y_Xuan, Zhang Jun, Zhang Ting, zhang wenhui, Zhang Zheng, zhangchunle, Zhen Wang, zhiboniu, Zhong Hui, Zhou Wei, zhulei, zhupengyang, zlsh80826, 卖鱼的哲学, 石晓伟
