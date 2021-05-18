# Release Note

## Important Updates

The PaddlePaddle Framework V2.1.0 has the following important updates:

- Environment Adaptation: Add the support for Python 3.9, CUDA 11.2; Provide the support for [ROCm platform](https://rocmdocs.amd.com/en/latest/) (experimental); Provide the support for [Ascend AI processor](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/ascend-910) (experimental); Add the number of models that can run on [Baidu Kunlun chip](https://cloud.baidu.com/product/kunlun.html) . For details, please see: [Getting Started](https://www.paddlepaddle.org.cn/install/quick).

- Distributed training: Support training extremely large models with 4D parallelism, powered by data parallelism, model parallelism, pipeline parallelism and arbitrary sharding, in addition, they can combine with AMP and the new ReCompute strategy to achieve better efficiency. 

- Framework function: Complete a number of enhancements and performance optimizations, in particular, including the following important new functions:
  
  - Provide a new solution for customizing operators outside the framework, simplifying the process of writing custom operators and deploying training inference. For details see: [Customizing External Operators](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/07_new_op/new_custom_op_cn.html).
  - Add the inplace operation to reduce the memory consumption and improve performance, including View strategy, and 12 inplace APIs.
  - Add the high-level APIs to support mixed precision training; add `paddle.hub` to view, share, and load models.
  - Automatic mixed precision training optimization. Optimize the computational performance of multiple OPs in mixed precision training such as slice, where, range, etc., and improve the acceleration effect on MaskRCNN, ERNIE and other models.
  - oneDNN BF16 training: Enabled AMP (AutoMixedPrecision) pure_BF16 mode. Enabled BF16 SGD and initializers for less memory consumption. Enabled most of FWD & BWD BF16 ops for BF16 word2vec training.

- Paddle projects: For the latest updates to the official model libraries and suites of PaddlePaddle, please see: [Paddle projects notes along with PaddlePaddle2.1](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-projects-notes-along-with-PaddlePaddle2.1).

## Backwards Incompatible Changes

- The PaddlePaddle Framework 2.1 drops the support for python2 and python3.5. It is recommended that you upgrade your python to version 3.8 before using the PaddlePaddle. PaddlePaddle Framework 2.1 no longer provides the support for CUDA9 pre-built package. It is recommended that you upgrade the CUDA version before using the PaddlePaddle.
- The optimization of API visibility makes it impossible to import private APIs located in the deeply nested namespaces that are considered as implementation details by using `from deeply_nested_namespace import *`. It is recommended that you use the PaddlePaddle by following the instructions in the [API Documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html) on the PaddlePaddle website. Specifically, the following actions are no longer allowed in the PaddlePaddle Framework 2.1.

```python
# will import nothing from the deeply nested namespaces
from paddle.nn.layer.loss import *
from paddle.nn.layer.conv import *
```

- `Tensor.grad` Incompatible upgrade. The type of return value is changed from `numpy` to `Tensor`. ([#32142](https://github.com/PaddlePaddle/Paddle/pull/32142)) 

| 2.0                                                          | 2.1                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| import paddle<br /> x = paddle.to_tensor(5., stop_gradient=False)<br /> y = paddle.pow(x, 4.0)<br /> y.backward()<br /> type(x.grad)<br /> < class 'numpy.ndarray' >| import paddle<br /> x = paddle.to_tensor(5., stop_gradient=False)<br /> y = paddle.pow(x, 4.0)<br /> y.backward()<br /> type(x.grad)<br />< class 'paddle.Tensor' > |


- `paddle.jit.TraceLayer.save_inference_model` Interface incompatibility upgrade. Changed the original first parameter dirname to path, the name symbol is more generic and unified with interfaces such as paddle.save and load, indicating that the user specifies the prefix for saving the model path. ([#31989](https://github.com/PaddlePaddle/Paddle/pull/31989)) 

  | 2.0                                                          | 2.1                                                          |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | import os<br />import paddle<br />from paddle.vision.models import resnet18<br /><br />model = resnet18()<br />x = paddle.rand([1, 3, 224, 224])<br />_, static_layer = paddle.jit.TracedLayer.trace(model, input=[x])<br />save_path = './save_infer_model' <br />static_layer.save_inference_model(**dirname**=save_path) <br /><br />print(os.path.isdir(save_path))<br />print(len(os.listdir(save_path)))<br /><br /> True<br />205| import os<br />import paddle<br />from paddle.vision.models import resnet18<br /><br />model = resnet18()<br />x = paddle.rand([1, 3, 224, 224])<br />_, static_layer = paddle.jit.TracedLayer.trace(model, input=[x])<br />save_path = './save_infer_model' <br />static_layer.save_inference_model(**path**=save_path) <br /><br />print(os.path.isdir(save_path))<br />print([name for name in os.listdir('./') if name.startswith(save_path)])<br /><br /> False <br />`[save_infer_model.pdiparams]`|
  
  
- `paddle.io.DataLoader`return format incompatibility upgrade when user-define dataset only contains single field。If user-define dataset only contains single field and output data with code like `return image` or `yield image`，output data format in Paddle 2.0 is `[image_tensor]`，and output data format in Paddle 2.1 is `image_tensor` to keep data structure same with input.

  | 2.0                                                          | 2.1                                                          |
  | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | import numpy as np<br />import paddle<br />from paddle.io import DataLoader, Dataset<br /><br />class RandomDataset(Dataset):<br />def \_\_getitem\_\_(self, idx):<br />return np.random.random((2, 3)).astype('float32')<br /><br />def \_\_len\_\_(self): <br />return 10<br /><br />dataset = RandomDataset()<br />loader = DataLoader(dataset, batch_size=1)<br /> data = next(loader())<br /># data: [Tensor(shape=(1, 2, 3), dtype=float32)]|import numpy as np<br />import paddle<br />from paddle.io import DataLoader, Dataset<br /><br />class RandomDataset(Dataset):<br />def \_\_getitem\_\_(self, idx):<br />return np.random.random((2, 3)).astype('float32')<br /><br />def \_\_len\_\_(self): <br />return 10<br /><br />dataset = RandomDataset()<br />loader = DataLoader(dataset, batch_size=1)<br /> data = next(loader())<br /># data: Tensor(shape=(1, 2, 3), dtype=float32)|

## Training Framework

### Functional optimization (including distributed)

#### Basic API

- Add data types such as `paddle.dtype` and `paddle.float32` as data types within the Paddle. ([#32012](https://github.com/PaddlePaddle/Paddle/pull/32012))
- Add `paddle.nn.functional.glu`. ([#32096](https://github.com/PaddlePaddle/Paddle/pull/32096))
- Add `paddle.nn.utils.spectral_norm`. ([#32633](https://github.com/PaddlePaddle/Paddle/pull/32633))
- Add `paddle.Tensor.register_hook` API for registering the hook function for the gradient Tensor corresponding to the forward Tensor in dynamic graph scenes. ([#31775](https://github.com/PaddlePaddle/Paddle/pull/31775))
- Add the `Tensor.__array__` function to support `numpy.array(Tensor)` and `numpy.asarray(Tensor)` to convert `paddle.Tensor` type to `numpy.ndarray` type . ([#32300](https://github.com/PaddlePaddle/Paddle/pull/32300))
- Add the Tensor API: `Tensor.item(*args)`. It can convert the element at the specified position in Tensor to Python scalar value and return it. ([#32634](https://github.com/PaddlePaddle/Paddle/pull/32634))
- Add the `paddle.nn.LayerList` support for negative indexing. ([#31750](https://github.com/PaddlePaddle/Paddle/pull/31750))
- Add 12 dynamic graph inplace APIs: `clip_`, `scale_`, `add_`, `subtract_`, `ceil_`, `floor_`, `exp_`, `reciprocal_`, `round_`, `sqrt_`, `rsqrt_`, and `flatten_`. These inplace APIs cannot be called by using `paddle.api_` and should be called by using `Tensor.api_`. ([#32699](https://github.com/PaddlePaddle/Paddle/pull/32699))
- Add `paddle.autograd.backward` API for customizing the starting gradient. ([#31540](https://github.com/PaddlePaddle/Paddle/pull/31540))
- Add `paddle.nn.LayerDict` class. ([#31951](https://github.com/PaddlePaddle/Paddle/pull/31951))
- Add `layer.to` API. ([#32040](https://github.com/PaddlePaddle/Paddle/pull/32040))
- Add `paddle.autograd.PyLayer` API for supporting custom backward calculation of dynamic graphs on Python side. ([#32130](https://github.com/PaddlePaddle/Paddle/pull/32130))
- Add the support for `paddle.optimizer` to specify non-parametric Tensor as parameters for optimization in dynamic graphs. ([#32362](https://github.com/PaddlePaddle/Paddle/pull/32362))
- Add several `sequence*` functions in `paddle.static.nn`. Add `paddle.nn.functional` in `sequence_mask`. ([#32089](https://github.com/PaddlePaddle/Paddle/pull/32089))
- Add `paddle.nn.CTCLoss` parameters in `norm_by_times`. ([#32490](https://github.com/PaddlePaddle/Paddle/pull/32490))
- `paddle.fill_constant` supports `uint8_t`. ([#31911](https://github.com/PaddlePaddle/Paddle/pull/31911))
- `paddle.clip` supports `int32` and `int64`. ([#32373](https://github.com/PaddlePaddle/Paddle/pull/32373))
- Support the input data type to be int in Nearest neighbor mode in `paddle.nn.functional.interpolate`. ([#32270](https://github.com/PaddlePaddle/Paddle/pull/32270))
- All parameters in API that support passing in list or tuple are upgraded to support passing in list and tuple. ([#32344](https://github.com/PaddlePaddle/Paddle/pull/32344),  [#32528](https://github.com/PaddlePaddle/Paddle/pull/32528) [#32360](https://github.com/PaddlePaddle/Paddle/pull/32360))
- Optimize `softmax` operator performance. ([#31821](https://github.com/PaddlePaddle/Paddle/pull/31821))
- Optimize `paddle.norm` documentation description to clarify the functional differences between `paddle.norm` and `numpy.linalg.norm` API. ([#32530](https://github.com/PaddlePaddle/Paddle/pull/32530))
- Optimize the printing form of data type (`datatype`) of Tensor, for example, the `dtype` of Tensor of `float32` type is changed from `VarType.FP32` to `paddle.float32`. ([#30682](https://github.com/PaddlePaddle/Paddle/pull/30682))
- OneDNN Functional optimization
  - Upgraded oneDNN to 2.2.1 ([#31067](https://github.com/PaddlePaddle/Paddle/pull/31067) [#31473](https://github.com/PaddlePaddle/Paddle/pull/31473) [#30295](https://github.com/PaddlePaddle/Paddle/pull/30295) [32227](https://github.com/PaddlePaddle/Paddle/pull/32227))
  - Added more precise mkldnn kernel rules in GetExpectedKernelType based on kernel's data type. ([#29840](https://github.com/PaddlePaddle/Paddle/pull/29840))
  - Fused `layer_norm` subgraphs to single `layer_norm` op. ([#32162](https://github.com/PaddlePaddle/Paddle/pull/32162), [#30891](https://github.com/PaddlePaddle/Paddle/pull/30891), [#30962](https://github.com/PaddlePaddle/Paddle/pull/30962))
  - Reduced unnecessary memory allocation during creation of `elementwise_mul` operator ([#30203](https://github.com/PaddlePaddle/Paddle/pull/30203))
  - Improved memory consumption used in cache per thread ([#30358](https://github.com/PaddlePaddle/Paddle/pull/30358))
  - Added oneDNN FP32 and INT8 support for vanilla LSTM ([#30719](https://github.com/PaddlePaddle/Paddle/pull/30719) [#31894](https://github.com/PaddlePaddle/Paddle/pull/31894))
  - Added OneDNN `hardswish` support ([#30211](https://github.com/PaddlePaddle/Paddle/pull/30211))
  - Added `bilinear_interp_v2` and `nearest_interp_v2` oneDNN FP32 kernels ([#32312](https://github.com/PaddlePaddle/Paddle/pull/32312))
- Updated Xbyak to v5.81 ([#30809](https://github.com/PaddlePaddle/Paddle/pull/30809))
- Fix `paddle.io.DataLoader` to support data sets containing nested complex data formats such as list, dict and string, and fix the occasional error report and unreleased resources when the program exits during the iteration. ([#31481](https://github.com/PaddlePaddle/Paddle/pull/31481))
- Fix the problem caused by modifying the root logger of logging library in paddle. ([#32706](https://github.com/PaddlePaddle/Paddle/pull/32706))
- Fix the problem of `L1Decay` error report in `backward` dynamic graph mode. ([#32718](https://github.com/PaddlePaddle/Paddle/pull/32718))
- Fix the problem that nan comes out in setting `ignore_index` and `reduction='mean'` in `paddle.nn.functional.cross_entropy`. ([#32545](https://github.com/PaddlePaddle/Paddle/pull/32545))
- Fix the problem that the output type is bool during the summing of bool tensor and float tensor. ([#32272](https://github.com/PaddlePaddle/Paddle/pull/32272))
- Fix the calculation error of comparison class API in broadcast. ([#32470](https://github.com/PaddlePaddle/Paddle/pull/32470))
- Fix the gradient calculation error under broadcast where right input is large shape in addition, subtraction, multiplication and division. ([#30818](https://github.com/PaddlePaddle/Paddle/pull/30818))
- Fix the problem of the calculation result of segment mean OP being incorrect when processing the large shape tensor input. ([#32610](https://github.com/PaddlePaddle/Paddle/pull/32610))
- Fix the problem of the data type of optimizer variables not matching with the data type of model parameters. ([#29917](https://github.com/PaddlePaddle/Paddle/pull/29917))
- Fix the error report in `num worker>0` when the `paddle.io.DataLoader` pre-processing includes the paddle operation. ([#31177](https://github.com/PaddlePaddle/Paddle/pull/31177))
- Fix the error report when printing empty tensor. ([#32501](https://github.com/PaddlePaddle/Paddle/pull/32501))
- Adjust the initialization order of static graph parameters, and keep consistency with dynamic graphs after adjustment, so that the same model is set with the same random seed to get the same parameters initialized in dynamic graphs and static graphs. ([#32177](https://github.com/PaddlePaddle/Paddle/pull/32177))
- Fix the bug that `paddle.to_tensor` does not support accepting `dtype=Tensor.dtype`. ([#31931](https://github.com/PaddlePaddle/Paddle/pull/31931))
- Fix the bug that the gradient is nan when 2 inputs are equal in `paddle.dist`. ([#32448](https://github.com/PaddlePaddle/Paddle/pull/32448))
- `paddle.nn.functional.temporal_shift` added `data_format` property to support to set to NCHW or NHWC. ([#31642](https://github.com/PaddlePaddle/Paddle/pull/31642))
- Fix the problem of the calculation result being incorrect in `adaptive_avg_pool2d` when the input data type is float16. ([#31887](https://github.com/PaddlePaddle/Paddle/pull/31887))
- `paddle.nn.Layer.sublayers` and `paddle.nn.Layer.named_sublayers`: Modify the `include_sublayers = True` parameter of original `paddle.nn.Layer.sublayers` to `include_self = False`, thus fixing the problem of returning null of the former `include_sublayers = False`. Now the default behavior is the same as that when no parameter is filled in, that is, return all recursive sublevels that don't contain themselves. When `include_self = True` is the same as the literal meaning, return all recursive sublevels that contain themselves. The `include_sublayers` parameter in `paddle.nn.Layer.named_sublayers` is directly removed. Other behaviors remain unchanged. ([#31824](https://github.com/PaddlePaddle/Paddle/pull/31824) )

| 2.0                                                          | 2.1                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| from paddle.vision.models import resnet18<br/>model = resnet18()<br/><br/>print(len(model.sublayers(include_sublayers=True)))<br/>print(len(model.sublayers(include_sublayers=False)))<br/><br/>67<br/>0<br/> | from paddle.vision.models import resnet18<br/>model = resnet18()<br/><br/>print(len(model.sublayers(include_self=True)))<br/>print(len(model.sublayers(include_self=False)))<br/><br/>68<br/>67<br/> |

#### High-level API

- Add the `paddle.hub` function. Provide `help`, `list` and `load` functions for viewing and loading third-party models, and support the loading of remote and local repository. ([#31873](https://github.com/PaddlePaddle/Paddle/pull/31873))
- Support the mixed precision training. Provide O0, O1, O2 three modes, which correspond to FP32 training, automatic mixed precision training, pure FP16 training respectively. At present, pure FP16 training only supports static graphs. ([#31417](https://github.com/PaddlePaddle/Paddle/pull/31417))
- Support the image transformation of the `paddle.Tensor` type, including operators such as `normalize, to_grayscale, vflip, hflip, crop, center_crop, pad, rotate, resize`. ([#32705](https://github.com/PaddlePaddle/Paddle/pull/32705))

#### Dynamic Graphs to Static Graphs

Fix the bug of dynamic graphs converted to static graphs.

- The shape returned by the static graph `arange、range` API is not consistent with the dynamic graph.
- `paddle.to_tensor` supports the input as `int，float，bool` basic type in dynamic to static.
- Support the parsing of the dict derivative syntax in the for loop. ([#32159](https://github.com/PaddlePaddle/Paddle/pull/32159))
- Fix the problem of undeclared variables errors in the nested control flow statements in some scenarios. ([#32153](https://github.com/PaddlePaddle/Paddle/pull/32153))
- Fix the bug that the float16 type is missed in `expand` op. ([#32238](https://github.com/PaddlePaddle/Paddle/pull/32238))
- Fix the bug of returning the gradient information as None when the shape dimension is 6 in the `expand_v2、tile、expand、expand_as、expand_as_v2、meshgrid` 6 OP backward gradient solution. ([#32004](https://github.com/PaddlePaddle/Paddle/pull/32004))
- Fix the problem that the `paddle.jit.TraceLayer.save_inference_model` interface is inconsistent with `paddle.static.load_inference_model` because the network structure and parameters are not saved at the same time. ([#31989](https://github.com/PaddlePaddle/Paddle/pull/31989))

#### Mixed Precision Training

- The op that does not support fp16 kernel is automatically kept as fp32 calculation in the dynamic graph mixed precision interface auto\_cast. ([#32543](https://github.com/PaddlePaddle/Paddle/pull/32543))
- Fix the unexpected error in the static graph mixed precision training caused by the incomplete statistics of the Op list (`unsupported_fp16_list`) which does not support FP16 calculation. The list of Op that currently does not support FP16 calculation can be generated automatically according to the runtime environment. ([#32102](https://github.com/PaddlePaddle/Paddle/pull/32102))
- In the for loop in the `update_loss_scaling`, optimize the problem that multiple identical cuda kernel are fused into one cuda kernel. ([#32554](https://github.com/PaddlePaddle/Paddle/pull/32554))
- Optimize the slow performance in `slice` multi-dimensional cases. ([#32266](https://github.com/PaddlePaddle/Paddle/pull/32266))
- Optimize the redundant copy problem when `elementwise_add_grad` inputs and outputs are the same. ([#32051](https://github.com/PaddlePaddle/Paddle/pull/32051))
- In the for loop in the `check_finite_and_unscale`, optimize the problem that multiple identical cuda kernel are fused into one cuda kernel. ([#31954](https://github.com/PaddlePaddle/Paddle/pull/31954))
- Optimize the `range` parameter redundant copy problem. ([#30811](https://github.com/PaddlePaddle/Paddle/pull/30811))
- Optimize the slow performance problem of  `top_k_v2` in `input_width <= 1024`. ([#30403](https://github.com/PaddlePaddle/Paddle/pull/30403))
- Migrate `where_index` CPU calculation process to GPU for completion. ([#30601](https://github.com/PaddlePaddle/Paddle/pull/30601))

#### BF16 Training 

- Added initial bf16 amp integration that modify models by adding cast ops to BF16 enabled ops in the forward pass. [#31093](https://github.com/PaddlePaddle/Paddle/pull/31093) 
- Added BF16 pure_mode, which means adding support for BF16 training based on BF16-enabled ops list and enable BF16 parameters, BF16 operators, BF16 decorator for optimizer during training.  [#32281](https://github.com/PaddlePaddle/Paddle/pull/32281) [#32681](https://github.com/PaddlePaddle/Paddle/pull/32681)
- Added CPU core flags verification for BF16 fast performance support. [#30551](https://github.com/PaddlePaddle/Paddle/pull/30551)
- Unification of BF16 enablement process [#31034](https://github.com/PaddlePaddle/Paddle/pull/31034)
- Added BF16 Constant Initializer and for other initializers, add cast op to convert other initializer output to be BF16 datatype. [#31935](https://github.com/PaddlePaddle/Paddle/pull/31935) 
- Added BF16 uniform random initializer [#32468](https://github.com/PaddlePaddle/Paddle/pull/32468)
- Added mechanism that converts startup_program initializers to BF16 [#32720](https://github.com/PaddlePaddle/Paddle/pull/32720)
- Added BF16 support for sgd operator CPU kernel. [#32162](https://github.com/PaddlePaddle/Paddle/pull/32162) 
- Added BF16 support for lookup_table operator. [#31558](https://github.com/PaddlePaddle/Paddle/pull/31558)
- Added Sum kernel for CPU supporting BF16 and SelectedRows [#32755](https://github.com/PaddlePaddle/Paddle/pull/32755) [#32631](https://github.com/PaddlePaddle/Paddle/pull/32631) 
- Added Conv Transpose BF16 support [#30877](https://github.com/PaddlePaddle/Paddle/pull/30877) 
- Added elementwise_add bf16 grad [#30925](https://github.com/PaddlePaddle/Paddle/pull/30925)
- Added reshape op BWD grad bf16 [#31035](https://github.com/PaddlePaddle/Paddle/pull/31035)
- Added broadcasting support in elementwise_add grad bf16/fp32 [#31385](https://github.com/PaddlePaddle/Paddle/pull/31385) 
- Added Elementwise Mul grad fp32/bf16 [#31647](https://github.com/PaddlePaddle/Paddle/pull/31647)
- Added LSTM BF16 and fixed GRU BF16 [#31234](https://github.com/PaddlePaddle/Paddle/pull/31234) 
- Added oneDNN reduce_op fp32 and bf16 kernels [#31816](https://github.com/PaddlePaddle/Paddle/pull/31816) 
- Added oneDNN reduce_op GRAD fp32 and bf16 kernels [#32280](https://github.com/PaddlePaddle/Paddle/pull/32280) [#32592](https://github.com/PaddlePaddle/Paddle/pull/32592) 

#### Distributed Training Optimization

- New graph-based retrieval engine for training distributed graph neural network over trillion edges([#31226](https://github.com/PaddlePaddle/Paddle/pull/31226)).
- Added index-based data sampling class to support sampling from graph and TDM/OTM tree([#31696](https://github.com/PaddlePaddle/Paddle/pull/31696)).
- Added `paddle.distributed.send, paddle.distributed.recv` to improve the distributed communication API. ([#32504](https://github.com/PaddlePaddle/Paddle/pull/32504))
- Added `paddle.distributed.new_group` and `paddle.distributed.wait` for general communication API. ([#31682](https://github.com/PaddlePaddle/Paddle/pull/31682))
- Support to initialize the  `sync_parameters_buffer`in the distributed dynamic graph, which solved the issue that the buffer of the dynamic graph is not globally initialized. ([#31625](https://github.com/PaddlePaddle/Paddle/pull/31625))
- \[Hybrid Parallel] Support 4D Hybrid Parallelism: Data Parallelism/Sharding/Pipeline Parallelism/Model Parallelism with Fleet static graph mode(([#32486](https://github.com/PaddlePaddle/Paddle/pull/32486) [#32485](https://github.com/PaddlePaddle/Paddle/pull/32485) [#31996](https://github.com/PaddlePaddle/Paddle/pull/31996) [#31939](https://github.com/PaddlePaddle/Paddle/pull/31939) [#31796](https://github.com/PaddlePaddle/Paddle/pull/31796)).
- Pipeline Parallelism supports 1F1B scheduling method to optimize the memory usage of GPU. Theoretically, it is constant([#31786](https://github.com/PaddlePaddle/Paddle/pull/31786)).
- \[Hybrid Parallel] Sharding strategy optimization: support Gradients aggregation, reducing the amount of parameter communication, and improving the speed of training. ([#31884](https://github.com/PaddlePaddle/Paddle/pull/31884))
- \[Hybrid Parallel] Added optimizer state offload in the Sharding strategy, to reduce the memory usage of GPU. ([#32134](https://github.com/PaddlePaddle/Paddle/pull/32134))
- \[Hybrid Parallel] Support the persistence of the broadcast ID’s socket service, reduced the conflicts of ports in the hybrid parallelism. ([#31589](https://github.com/PaddlePaddle/Paddle/pull/31589))
- \[Parameter Server] Optimize the output and printing of LOG, and remove invalid logs.
- \[Parameter Server] Optimize the sparse parameter storage structure, with large memory reduction for small dimensions (below 64).
- \[Parameter Server] Fix the bug of access policy taking effect in the distributed prediction.
- HeterPs supports multiple machines.

##### Hybrid Parallelism with dynamic Graph

Support hybrid parallelism in the distributed dynamic graph mode, powered by data parallelism, model parallelism and pipeline parallelism, in addition, they can combine with AMP and the new ReCompute strategy to achieve better efficiency.

- Support hybrid parallelism with the Fleet dynamic graph API, and any arbitrary combination of data/model/pipeline parallelism. ([#32248](https://github.com/PaddlePaddle/Paddle/pull/32248))
-  Added parameter `find_unused_parameters` n the data parallelism of distributed dynamic graph to support grouping control flow in the network. ([#31625](https://github.com/PaddlePaddle/Paddle/pull/31625))
- Added `VocabParallelEmbedding`, `ColumnParallelLinear`, `RowParallelLinear` Fleet API for model parallelism. Added `model_parallel_random_seed`/`get_rng_state_tracker` for the random control used in model parallelism. ([#32248](https://github.com/PaddlePaddle/Paddle/pull/32248))
- Added `distributed_scaler` interface for loss scaler of AMP combined with the hybrid parallelism strategy. ([#32354](https://github.com/PaddlePaddle/Paddle/pull/32354))
- Added `PipelineLyaer` to partition graph in the pipeline parallelism, added `LayerDesc` or description of dynamic graph Layer to reduce memory initialization. ([#32449](https://github.com/PaddlePaddle/Paddle/pull/32449))
- Add Recompute strategy for dynamic graphs. ([#32516](https://github.com/PaddlePaddle/Paddle/pull/32516))

#### Custom OP

- Add support for using custom OP function on Mac platform. ([#31976](https://github.com/PaddlePaddle/Paddle/pull/31976))
- Support automatic search function of C++/v11 header file directory on Mac platform, compatible with the situation that multiple versions of clang may exist locally.
- Add support for Op forward/backward function Attribute parameter, inferShape, and InferDtype function input parameter using the `const &` type. ([#31588](https://github.com/PaddlePaddle/Paddle/pull/31588))
- Add support for using three framework internal data types `paddle::complex64, paddle::complex128, paddle::float16`  in the implementation of custom Op. ([#31602](https://github.com/PaddlePaddle/Paddle/pull/31602), [#31657](https://github.com/PaddlePaddle/Paddle/pull/31657), [#31669](https://github.com/PaddlePaddle/Paddle/pull/31669), [#31725](https://github.com/PaddlePaddle/Paddle/pull/31725))
- Add support for using `std::vector<paddle::Tensor>` type parameters as input of forward/backward functions in custom Op. ([#31535](https://github.com/PaddlePaddle/Paddle/pull/31535))
- Add support for the InferShape function using Attribute parameter as input. ([#31713](https://github.com/PaddlePaddle/Paddle/pull/31713))
- Optimize the call stack of auto-generated Python API under dynamic graph to improve the execution efficiency. ([#32209](https://github.com/PaddlePaddle/Paddle/pull/32209))
- Reduce the error reporting condition when checking the compiler cl.exe on Windows, and enhance the self-test robustness in Windows environment. ([#32769](https://github.com/PaddlePaddle/Paddle/pull/32769))
- Fix a bug in compiler selection when installing multiple CUDA environments on Windows. ([#31694](https://github.com/PaddlePaddle/Paddle/pull/31694))
- Fix a bug in Python encoding issue when installing Chinese version of VS on Windows. ([#31493](https://github.com/PaddlePaddle/Paddle/pull/31493))
- Remove the dependency on separate dynamic library files and link only the framework core dynamic library files. ([#32404](https://github.com/PaddlePaddle/Paddle/pull/32404)、[#32769](https://github.com/PaddlePaddle/Paddle/pull/32769))
- Remove the previous old custom OP scheme and clean up the redundant library files and header files in the whl package, reducing the whl package size by about 11M. ([#31813](https://github.com/PaddlePaddle/Paddle/pull/31813)), ([#32463](https://github.com/PaddlePaddle/Paddle/pull/32463))

#### Model saving and loading

- `paddle.save, paddle.load` supports saving and loading of Tensor. ([#31756](https://github.com/PaddlePaddle/Paddle/pull/31756))
- `paddle.save, paddle.load` supports saving and loading of `list[Tensor]、dict[Tensor]、tuple[Tensor]` and `list、tuple、dict` nested structures containing Tensor. ([#32446](https://github.com/PaddlePaddle/Paddle/pull/32446))
- `paddle.save, paddle.load` supports saving and loading of Layer. ([#32446](https://github.com/PaddlePaddle/Paddle/pull/32446))
- `paddle.save, paddle.load` supports saving and loading of Program. ([#32336](https://github.com/PaddlePaddle/Paddle/pull/32336))
- `paddle.save, paddle.load` supports saving and loading of single Tensor in C++ binary format. ([#32211](https://github.com/PaddlePaddle/Paddle/pull/32211))
- `paddle.jit.save, paddle.jit.load` supports saving and loading of Fucntion without parameters. ([#32430](https://github.com/PaddlePaddle/Paddle/pull/32430))

### Performance optimization (including distributed)

- Optimize key operators to improve single GPU training performance of multiple models. Deeplabv3+ single card FP32 and AMP performance are improved by 11% and 72% respectively. TSM single card AMP performance is improved by 44.5%. HRNet single card FP32 and AMP are improved by 46% and 51% respectively.
- Add `index_sample` CUDA implementation. ([#30380](https://github.com/PaddlePaddle/Paddle/pull/30380))
- Implement the CUDA Kernel of `relu, leaky_relu` operator, replacing the original Eigen implementation, with a total improvement of 5% - 20% in forward and backward directions. ([#31869](https://github.com/PaddlePaddle/Paddle/pull/31869), [#31841](https://github.com/PaddlePaddle/Paddle/pull/31841))
- `temporal_shift` Performance improvement by 20% to 40%. ([#31642](https://github.com/PaddlePaddle/Paddle/pull/31642))
- Optimize `depthwise_conv2d`. Performance is improved by 30% to 50% under the NHWC format. ([#31667](https://github.com/PaddlePaddle/Paddle/pull/31677))
- Optimize `interp_bilinear_grad` operator NCHW performance with improvement by 19% - 303%. ([#30950](https://github.com/PaddlePaddle/Paddle/pull/30950))
- Optimize the performance of `adaptive_avg_pool2d` operator NCHW. In case of output\_size = 1 case, improve by 80%~90%. ([#31197](https://github.com/PaddlePaddle/Paddle/pull/31197))
- In conv op, when dtype is float16, forward and backward support the enabling of `exhaustive_search`. ([#30959](https://github.com/PaddlePaddle/Paddle/pull/30959))
- When `weight_decay` parameter of `momentum` is set to float type, the fusion of  `momentum` and `L2Decay` is achieved ([#30881](https://github.com/PaddlePaddle/Paddle/pull/30881))
- Implement CUDA Kernel when `log_softmax` operator `axis` is the last dimension and dimension is equal to or smaller than 1024. Compared to the original Eigen, the forward and backward operator performance is improved by 4.55x ~ 26.45x. ([#31630](https://github.com/PaddlePaddle/Paddle/pull/31630), [#32180](https://github.com/PaddlePaddle/Paddle/pull/32180))

## Inference Deployment

### Model Quantization

- Add the support for saving FP32 model as FP16 model. ([#32112](https://github.com/PaddlePaddle/Paddle/pull/32112))
- Refactor the module of statistical output quantization information in dynamic graph quantization training to support multi-Block and multi-branch models to enhance generality. ([#31680](https://github.com/PaddlePaddle/Paddle/pull/31680) [#31710](https://github.com/PaddlePaddle/Paddle/pull/31710) [#31784](https://github.com/PaddlePaddle/Paddle/pull/31784) [#31861](https://github.com/PaddlePaddle/Paddle/pull/31861))
- Dynamic graph quantization training function supports the skipping of the quantization OP and forms the successful connection at the prediction side. ([#31704](https://github.com/PaddlePaddle/Paddle/pull/31704))

### Paddle Inference

#### Function Upgrade

- Release C API (experimental).  The function of  new C API is basically equal to that of C + +. ([#32225](https://github.com/PaddlePaddle/Paddle/pull/32225))
- The prediction framework python interface access to the train custom operators. After loading a custom operator during training, users can execute the deployment of prediction models containing this custom operator directly through PaddlePredictor, just like the framework's native operator. ([#32533](https://github.com/PaddlePaddle/Paddle/pull/32533))
- The underlying implementation of Tensor has been refactored internally to decouple from the old ZeroCopyTensor data structure. This upgrade does not involve user API changes and is transparent to users. ([#31402](https://github.com/PaddlePaddle/Paddle/pull/31402))
- Support TensorRT serialization and deserialization when loading models from memory. ([#31342](https://github.com/PaddlePaddle/Paddle/pull/31342))

#### Performance Optimization

- Support quantilized  ERNIE models to be inferred with  the mixed precision using TensorRT, where Matmul is computed with Int8 precision and other parts are computed with FP16 precision. Compared with the pure FP16 inference, the inference performance of the standard ERNIE model on XNLI dataset is improved from 1898 seq/s to 2310 seq/s at batch size=40 on T4, improving by 17.8%. ([#32232](https://github.com/PaddlePaddle/Paddle/pull/32232))

#### Ease-of-use optimization

- Add  error messages when the user enables the TensorRT variable-length input settings, and the wrong input shape  is provided. ([#32155](https://github.com/PaddlePaddle/Paddle/pull/32155))
- Add runtime TensorRT version check. If the major version number of TensorRT at runtime and compile time differs, the warning is generated. ([#32443](https://github.com/PaddlePaddle/Paddle/pull/32443))
- Add the TensorRT VERBOSE level log switch. Users can enable the TensorRT VERBOSE log by `export GLOG_v=3` to print more debugging information. ([#32459](https://github.com/PaddlePaddle/Paddle/pull/32459))

#### BugFix
- Fix the error of insufficient graphics card or video memory of unspecified usage at the end of prediction. ([#32655](https://github.com/PaddlePaddle/Paddle/pull/32655))
- Fix the CPU performance issue caused by informal values of native inference in dynamic graphs. ([#32350](https://github.com/PaddlePaddle/Paddle/pull/32350))
- Fix the problem of requiring the setting of the calibration table path in the data read from memory when TensorRT inference is enabled by using the PaddleSlim quantization model. ([#32676](https://github.com/PaddlePaddle/Paddle/pull/32676))
- Upgrade the TensorRT quantization calibration table interface, fix the problem that TensorRT offline quantization is not supported on DLA. ([#31060](https://github.com/PaddlePaddle/Paddle/pull/31060))
-  Fix the problem of the number of header of crop Attention not being supported when using variable length method for ERNIE/BERT model inference (EnableTensorRtOSS). ([#31497](https://github.com/PaddlePaddle/Paddle/pull/31497))
- Fix the occasional diff problem caused by the instable QK input sequence of the BERT model trained after version 2.0 ([#32659](https://github.com/PaddlePaddle/Paddle/pull/32659))
- Fix the problem that ERNIE model reports an error or incorrect result due to the wrong order of input variable names when TensorRT varlen acceleration is enabled. ([#32482](https://github.com/PaddlePaddle/Paddle/pull/32482))
- Fix the bug that plugin ElementwisePluginDynamic serialization of TensorRT fails. ([#31587](https://github.com/PaddlePaddle/Paddle/pull/31587))
- Fix the problem of subsequent OP dimension error caused by FC layer dimension complement 1 under TensorRT dynamic shape. ([#32458](https://github.com/PaddlePaddle/Paddle/pull/32458), [#31803](https://github.com/PaddlePaddle/Paddle/pull/31803))
- Fix the problem of `repeated_fc_relu_fuse_pass.cc` error when FC uses Padding. ([#32648](https://github.com/PaddlePaddle/Paddle/pull/32648/files))
- Fix the problem of the result of conv2d\_transpose op being wrong when using TensorRT inference. ([#32593](https://github.com/PaddlePaddle/Paddle/pull/32593))
- Fix the problem with OCR INT8 model oneDNN prediction errors caused by incorrect comparison of NAN. ([#32227](https://github.com/PaddlePaddle/Paddle/pull/32227))
- Fix the problem of data contention when deploying multiple models for oneDNN prediction on multiple executors with multiple threads. ([#32499](https://github.com/PaddlePaddle/Paddle/pull/32499),  [#32136](https://github.com/PaddlePaddle/Paddle/pull/32136) [#32664](https://github.com/PaddlePaddle/Paddle/pull/32664))


## Environment Adaptation

### Compile and install

- Add support for CUDA11.2 compilation. Support the compilation based on the 3070/3080/3090 graphics card architecture. ([#31529](https://github.com/PaddlePaddle/Paddle/pull/31529))
- Add the support for compilation of Windows Visual Studio 2017. Upgrade all supporting facilities such as release, CI/CE, compilation documentation, etc. from VS2015 to VS2017 comprehensively. ([#311652](https://github.com/PaddlePaddle/Paddle/pull/31652))
- Add support for cuda11.2 image. ([#32531](https://github.com/PaddlePaddle/Paddle/pull/32531))
- cuda10.1 image support for gcc 5.4. ([#32531](https://github.com/PaddlePaddle/Paddle/pull/32531))
- Add support for python 3.9 in mirrors. ([#32385](https://github.com/PaddlePaddle/Paddle/pull/32385))
- Fix the bug of `run_check` interface, and add the check of dynamic graph in  `run_check` interface: Now the logic of `run_check` detecting paddle installation first detects whether there is a GPU on the user's machine. If not, report warning, without considering the users who install the cpu package ([#32428](https://github.com/PaddlePaddle/Paddle/pull/32428))
- Fix the problem of lack of symlink method on Windows system. ([#31006](https://github.com/PaddlePaddle/Paddle/pull/31006))

### New hardware training support

- Add the support for Hygon chips:  PaddlePaddle, based on ROCM version 4.0.1, can train and infer models on Hygon CPU and DCU. A total of 36 models of 7 categories of image classification, target detection, image segmentation, natural language processing, recommendation systems, video classification and speech synthesis have been validated.
- Add the support of Ascend chips: support for single hosting, multiple accelerators training on Ascend NPUs.
- Kunlun hardware training support
  - Kunlun XPU supports dynamic graph distributed training. ([#30455](https://github.com/PaddlePaddle/Paddle/pull/30455),  [#30671](https://github.com/PaddlePaddle/Paddle/pull/30671))
  - Kunlun XPU supports fleet distributed training. ([#30858](https://github.com/PaddlePaddle/Paddle/pull/30858))
  - Kunlun XPU supports spawn to start multi-card training and optimize XPU dynamic graph multi-card performance. ([#31130](https://github.com/PaddlePaddle/Paddle/pull/31130))
  - Kunlun XPU static graph multi-card supports the optimization of fuse allreduce and gradient merge. ([#31104](https://github.com/PaddlePaddle/Paddle/pull/31104))
  - Support Kunlun XPU in the exposure of all\_reduce/reduce collection communication API. ([#32303](https://github.com/PaddlePaddle/Paddle/pull/32302))
  - Fix the bug of the random hang of Kunlun XPU dynamic graph multi-card. ([#32662](https://github.com/PaddlePaddle/Paddle/pull/32662))

## Thanks to our Contributors

This release contains contributions from:

123malin, Adam Osewski, alncat,  arlesniak, AshburnLee, Aurelius84, Bai Yifan, Baibaifan, Bin Lu, cc, ceci3, chajchaj, chalsliu, channings, Chen Long, Chen Weihang, chen zhiyu, Chengmo, chentianyu03, cnn, CtfGo, cucuzg, danleifeng, denglin-github, Double\_V, fangshuixun007, Feiyu Chan, fluffyrita, FlyingQianMM, FNRE, furnace, GaoWei8, GeminiCarrie, gongweibao, Gradie, GT-Zhang, Guanghua Yu, Guo Sheng, guofei, hong, houj04, huangjun12, huangxu96, Huihuang Zheng, hutuxian, iducn, Jacek Czaja, Jack Zhou, jakpiase, JamesLim, Jiabin Yang, jiangcheng, Jiaqi Liu, Jiawei Wang, joanna.wozna.intel, joejiong, JZ-LIANG, Kaipeng Deng, Kqnonrime, kuizhiqing, Lei.C, Leo Chen, lidanqing, LielinJiang, lijianshe02, lilong12, limingshu, littletomatodonkey, liu zhengxi, LiuChiachi, liuyuhui, liym27, LoveAn, LutaoChu, minghaoBD, mls1999725, niuliling123, Ouyang Chao, pangyoki, parap1uie-s, Pei Yang, procr, Qi Li, qingqing01, QingshuChen, Ren Wei (任卫), ronnywang, ruri, seemingwang, Shang Zhizhou, shanliang1992, ShenLiang, Shibo Tao, Steffy-zxf, syyxsxx, taixiurong, tangwei12, Tao Luo, Thomas Young, Thunderbrook, tianshuo78520a, TTerror, wangchaochaohu, wangguanzhong, wanghuancoder, wangna11BD, WangXi, wangxinxin08, wawltor, Wei Shengyu, weihaoji, WeiXin, wenbin, Wenyu, whs, Wilber, winter-wang, Wojciech Uss, wuhuanzhou, wuyefeilin, XGZhang, XiangGao, XiaoguangHu, xiaoting, xiegegege, xiemoyuan, xingfeng01, Yang Zhang, yaoxuefeng, yiak, yingshengBD, yinhaofeng, Yiqun Liu, ykkk2333, yongqiangma, Yuang Liu, yukavio, YUNSHEN XIE, Y_Xuan, Zhang Jun, Zhang Ting, zhang wenhui, Zhang Zheng, zhangchunle, Zhen Wang, zhiboniu, Zhong Hui, Zhou Wei, zhulei, zhupengyang, zlsh80826, 卖鱼的哲学, 石晓伟
