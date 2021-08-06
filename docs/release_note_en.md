
# 2.1.2 Release Note


## Important Updates

This release mainly fixes some features and performance issues in 2.1.1.  See the following highlights:

-  Fix several known problems with dynamic to static syntax transcriptions
-  C++ version check upgrade from C++11 to C++14 during Custom OP compile

## Training framework

### Functional optimization (including distributed)

#### Basic API

- Restore the callability of the `paddle.vision.xxx` path module to enhance compatibility of the old versions; yet it is not recommended to use this path to call the API([#34489](https://github.com/PaddlePaddle/Paddle/pull/34489))
- Fix `paddle.concat` overflow when applied to multiple Tensor with large `shape` ([#34396](https://github.com/PaddlePaddle/Paddle/pull/34396))
- `paddle.flip` supports input axis as integer, and improves performance in dynamic graph mode ([#34477](https://github.com/PaddlePaddle/Paddle/pull/34477))
- Fix `paddle.slice` out-of-bounds access problem when input and output addresses are the same([#34265](https://github.com/PaddlePaddle/Paddle/pull/34265))
- Fix the problem of wrong order of input parameters of `paddle.nn.Unfold`([#34251](https://github.com/PaddlePaddle/Paddle/pull/34251))
- Add several interfaces for `Tensor` under static graphs such as `size(), detach()`, etc. ([#33330](https://github.com/PaddlePaddle/Paddle/pull/33330))
- Add incompatible upgrade note to the Warning content of `Tensor.grad`([#34262](https://github.com/PaddlePaddle/Paddle/pull/34264))
- Downlink `paddle.save` to save the function of  `Layer`([#34039](https://github.com/PaddlePaddle/Paddle/pull/34039))
- Fix `paddle.jit.save` for saving models on Mac systems that cannot be retrained on Linux platforms([#34154](https://github.com/PaddlePaddle/Paddle/pull/34154))
- Fix `layer_norm` with wrong `cuda kernel` parameters for large `size` input ([#33893](https://github.com/PaddlePaddle/Paddle/pull/33893))
- Fix `paddle.io.DataLoader` error reporting incompatible upgrade warning issue ([#34001](https://github.com/PaddlePaddle/Paddle/pull/34001))
- Fix `paddle.io.DataLoader` memory leak problem([#34301](https://github.com/PaddlePaddle/Paddle/pull/34301))

#### Dynamic to static map

- Add syntax support for nested use of `Sequential` container classes ([#34246](https://github.com/PaddlePaddle/Paddle/pull/34262))
- Add compatibility support for `Python3 type hint` syntax([#33745](https://github.com/PaddlePaddle/Paddle/pull/33745))
- Add support for non-`Tensor` types including `int, float, string, bool` in the `input_spec` argument of `@to_static`([#33464](https://github.com/PaddlePaddle/Paddle/pull/33464))
- Fix a number of known problems with the transcription of dynamic to static syntax ([#33963](https://github.com/PaddlePaddle/Paddle/pull/33963))

#### Custom OP

- C++ version check upgrade from C++11 to C++14 during Custom OP compile ([#30415](https://github.com/PaddlePaddle/Paddle/pull/34015)) 


## Inference Deployment


### Paddle Inference

#### Issue fix

- Fix wrong calculation result of  ERNIE model when `batch_size > 1` ([#33784](https://github.com/PaddlePaddle/Paddle/pull/33784))
- Fix the crash caused by splitting `TensortRT` inference path with right slash under windows.([#33885](https://github.com/PaddlePaddle/Paddle/pull/33885))
- Fix MKLDNN `elementwise` series OP's X does not support broadcast （[#33845](https://github.com/PaddlePaddle/Paddle/pull/33845)）

## Environment adaptation

### Compile and install

- Restrict the version range of dependent Gast libraries ( `gast>=0.3.3, <=0.4.0`)([#33850](https://github.com/PaddlePaddle/Paddle/pull/33850)) 
- Optimize `Avx/No-Avx` related installation error messages, reduce redundant Warning messages([#33885](https://github.com/PaddlePaddle/Paddle/pull/33905))

### New Hardware Adaptation

#### Kunlun hardware training support

- Modify the  `cmake` file of Kunlun to unify and update its operator library（[#34000](https://github.com/PaddlePaddle/Paddle/pull/34000)）

## Thanks to our Contributors

This release contains contributions from:

0x45f、Aurelius84、Chen Weihang、chentianyu03、HexToString、iducn、Jacek Czaja、Kaipeng Deng、Leo Chen、lzzyzlbb、Peihan、taixiurong、tianshuo78520a、WeiXin、wenbin、Wilber、wuhuachaocoding、xiongkun、Zhou Wei、 winter-wang .



# 2.1.1 Release Note

## Important Updates

This version fixed some function and performance issues of PaddlePaddle 2.1.0, and optimized some function. The important updates are as following:

- Optimize the API visibility of `paddle.distributed、paddle.device、paddle.vision` .
- Add support for dynamic conversion of user code for sublayer in the `paddle.nn.Sequential`.
- Add `SyncBatchNorm` support for AMP in dynamic graph, to improve the performance of dynamic graph `SyncBatchNorm` layer in AMP mode,

## Training Framework

### Functional optimization (including distributed)

#### Basic API

- Optimize the API visibility of `paddle.distributed、paddle.device、paddle.vision` , for more information, please see 2.1.0 Release Note. ([#33420](https://github.com/PaddlePaddle/Paddle/pull/32990))
- Add `paddle.is_compiled_with_rocm`. ([#33228](https://github.com/PaddlePaddle/Paddle/pull/33228))
- Add the `paddle.strided_slice` to  support bool type.（[#33373](https://github.com/PaddlePaddle/Paddle/pull/33373)）
- Add `paddle.equal_all、paddle.equal、paddle.greater_equal、paddle.greater_than、paddle.less_equal、paddle.less_than、paddle.not_equal` to support bool type. （[#33551](https://github.com/PaddlePaddle/Paddle/pull/33551)）
- Fix `paddle.utils.download` does not perform Retry when ConnectionError is abnormal.（[#33454](https://github.com/PaddlePaddle/Paddle/pull/33454)）
- Fix the issue of infershape error when `paddle.gather` axis is not equal to 0.（[#33553](https://github.com/PaddlePaddle/Paddle/pull/33553)）
- Fix segment fault caused by `paddle.io.DataLoader` when `num_workers=0` and `Dataset` returns GPU `Tensor` and sends it to `DataLoader` .（[#33487](https://github.com/PaddlePaddle/Paddle/pull/33487), [#33249](https://github.com/PaddlePaddle/Paddle/pull/33249)）
- Fix the issue that when use `slice` result as an lvalue of inplace operation, the error message of backward is not related to the error. （[#32981](https://github.com/PaddlePaddle/Paddle/pull/32981)）
- Fix the issue of `paddle.concat` support uint8 in dynamic graph.([#33667](https://github.com/PaddlePaddle/Paddle/pull/33667))
- Fix the issue of `paddle.grid_sample` GPU memory overflow and abnormal output. （[#33100](https://github.com/PaddlePaddle/Paddle/pull/33100)、[#33232](https://github.com/PaddlePaddle/Paddle/pull/33232)）
- Fix bug of roi_align, when the input width or height of rois is 0, the output feature should be 0 .（[#33446](https://github.com/PaddlePaddle/Paddle/pull/33446)）
- Fixed in some corner cases, input was modified to 'nan' bug of log_softmax op. （[#32937](https://github.com/PaddlePaddle/Paddle/pull/32937)）

#### Dynamic Graphs to Static Graphs

- Add support for dynamic conversion of user code for sublayer in the `paddle.nn.Sequential` .（[#33065](https://github.com/PaddlePaddle/Paddle/pull/33065)）
- Fix the issue of subscript syntax errors in the phase of static type analysis of variables in control flow for statement conversions. （[#32969](https://github.com/PaddlePaddle/Paddle/pull/32969)）
- Refactor the dynamic to static `param_guard` logic code to comprehensively solve the dynamic to static graph `Tensor` type conversion problem.（[#32985](https://github.com/PaddlePaddle/Paddle/pull/32985)）

#### Distributed Training

- Fix the error in `paddle.distributed.spawn` when using the default `nprocs` argument.（[#33249](https://github.com/PaddlePaddle/Paddle/pull/33249)）
- Fix the hang issue of training start caused by the inconsistent creation of pipeline parallel communication group.（[#32890](https://github.com/PaddlePaddle/Paddle/pull/32890)、[#33473](https://github.com/PaddlePaddle/Paddle/pull/33473)）
- Fix the issue of failed to save parameters in mixed parallelism.（[#33595](https://github.com/PaddlePaddle/Paddle/pull/33595)、[#33588](https://github.com/PaddlePaddle/Paddle/pull/33588)）
- Fix the issue that Fleet API cannot run `Program` directly.（[#33511](https://github.com/PaddlePaddle/Paddle/pull/33511)）
- Fix the hang issue caused by the uneven sample bucketing in the pure GPU training mode of heterogeneous parameter server.（[#32957](https://github.com/PaddlePaddle/Paddle/pull/32957)）

##### Hybrid Parallelism with Dynamic Graph

- Fix the the accuracy error of`TensorParallel`. Change the parameter initialization method of `TensorParallel` to ensure the randomness of the parameter after slicing.（[#33087](https://github.com/PaddlePaddle/Paddle/pull/33087)）
- Fix an accuracy error of `PipeLineParallel`. Fix the incorrect use of `microbatch` for `PipeLineParallel`.（[#33097](https://github.com/PaddlePaddle/Paddle/pull/33097)）
- Fix the issue that `new_group` API will hang when creating multiple communication groups.（[#33553](https://github.com/PaddlePaddle/Paddle/pull/33553)）

#### Mixed Precision Training

- Add `SyncBatchNorm` support for AMP in Dynamic graph, to improve the performance of dynamic graph `SyncBatchNorm` layer in AMP mode, and improve the 8-card AMP mode speedup ratio by 19% on `DeepLabV3P` model of [PaddleSeg].([#33709](https://github.com/PaddlePaddle/Paddle/pull/33709))

#### Custom OP

- Remove the dependency on `PADDLE_WITH_MKLDNN` macro for custom OP compilation.（[#32903](https://github.com/PaddlePaddle/Paddle/pull/32903)）
- Default setting `GLIBCXX_USE_CXX11_ABI=1` to resolve the issue of low GCC version that may cause compile-time errors.（[#33185](https://github.com/PaddlePaddle/Paddle/pull/33185)）
- Add support for c++14 syntax feature, and enable `-std=c++14` compile option by default. （[#33227](https://github.com/PaddlePaddle/Paddle/pull/33227)）

#### Others

- Fix the random segment error of training when `LoDTensorArray` is input of Op under multi-threading.（[#32984](https://github.com/PaddlePaddle/Paddle/pull/32984)）
- Fix an issue where parameter regularization is executed twice when both the regularizer of `paddle.ParamAttr` and the `weight_decay` of `paddle.optimize` are specified as `L2Decay`.（[#32881](https://github.com/PaddlePaddle/Paddle/pull/32881)）
- Fix the issue of corrupted characters of warning information in windows system.（[#33689](https://github.com/PaddlePaddle/Paddle/pull/33689)）

## Inference Deployment

### Model Quantification

- Fix the issue of skipping OP quantization in dynamic graph quantization training function.（[#32879](https://github.com/PaddlePaddle/Paddle/pull/32879)）
- Fix the issue that `layer_norm` does not save `out_threahold` attribute when quantized model is saved.（[#33610](https://github.com/PaddlePaddle/Paddle/pull/33707)）

### Paddle Inference

#### Function Upgrades

- Add converter/plugin of `gather_nd` 和 `reduce_sum` in Paddle-TRT.（[#33365](https://github.com/PaddlePaddle/Paddle/pull/33365)）
- Add `reshape` in Paddle-TRT.（[#33372](https://github.com/PaddlePaddle/Paddle/pull/33372)）

#### Performance Optimization

- Add the dynamic shape plugin of TensorRT` layer_norm` to improve model dynamic shape inference performance.（[#33448](https://github.com/PaddlePaddle/Paddle/pull/33448)）


#### 易用性优化

- Add Paddle Inference ROCm version of [Prediction Example Document](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/09_hardware_support/rocm_docs/infer_example_cn.html),  so as to add C++ prediction library version.txt with ROCm related version information.   ([#33290](https://github.com/PaddlePaddle/Paddle/pull/33290))
- Update XPU compilation options. Please refer to [#33581](https://github.com/PaddlePaddle/Paddle/pull/33581) for specific compilation options.

#### Bug Fixes

-  Fix the calculation error of `fused_fc_elementwise_layernorm` caused by too large number of threads under DCU. ([#33299](https://github.com/PaddlePaddle/Paddle/pull/33299))
-  Fix the issue that yolov3 model fails to run after gpu is turned on on nano and TX2.([#33442](https://github.com/PaddlePaddle/Paddle/pull/33442))
-  Fix the computation error when seq_len > 1024 in Paddle-TRT `multihead_matmul plugin` .（[#33365](https://github.com/PaddlePaddle/Paddle/pull/33365)）
-  Fix the incorrect output error caused by inconsistent order of input when ERNIE model becomes longer.（[#33622](https://github.com/PaddlePaddle/Paddle/pull/33622)）
-  Fix the reports error of OCR model in prediction on GPU.([#33431](https://github.com/PaddlePaddle/Paddle/pull/33431))
-  Fix the issue that `paddle.static.io.normalize_program` failed to export `paddle.static.normalize_program`.（[#33408](https://github.com/PaddlePaddle/Paddle/pull/33408)）
-  Fix the issue that conv with stride > 1 fails in TRT6.0 and below.([#33198](https://github.com/PaddlePaddle/Paddle/pull/33198) )
-  Fix the out-of-bounds error of GPU memory access when batch predicting images. ([#33370](https://github.com/PaddlePaddle/Paddle/pull/33370) )([#33531](https://github.com/PaddlePaddle/Paddle/pull/33531) )
-  Fix the issue of cache size setting failure on X86 CPU.  （[#33571](https://github.com/PaddlePaddle/Paddle/pull/33571)）
-  Fix TRT `conv2d_transpose op converter` dimension error setting. Now the model of `conv2d_transpose` op can work normally on TRT.（[#33242](https://github.com/PaddlePaddle/Paddle/pull/33242)）
-  Fix the error of prediction library compiled by sub-CUDA Arch on Jetson devices. This version will release the Jetson prediction library compiled by sub-Arch for users who have demand for shrinked prediction library binary size.（[#33269](https://github.com/PaddlePaddle/Paddle/pull/33269)）
-  Fix the issue that when using PaddleSlim quantitative model to load prediction from memory,  it still reports an error because the calibration table path is not set.（[#33629](https://github.com/PaddlePaddle/Paddle/pull/33629)）
-  Fix the issue that BERT/ERNIE gives wrong cuda error 400 when using TRT prediction on non-0 card.（[#33706](https://github.com/PaddlePaddle/Paddle/pull/33706)）
-  Fix a cmake syntax error caused by setting custom compilation parameters under Linux.（[#33621](https://github.com/PaddlePaddle/Paddle/pull/33621)）
-  Optimize the calculation accuracy of `layer_norm` and fix the problem of outputting Nan when input is large data. ([#33420](https://github.com/PaddlePaddle/Paddle/pull/33420))

## Environment Adaptation

### Compile and install

### Support of new hardware training

#### support of Kunlun chips

- Fix the `gather` op, add support of logsumexp op. ([#32931](https://github.com/PaddlePaddle/Paddle/pull/32931))

## Thanks to our Contributors

This release contains contributions from:
Aurelius84, cc, ceci3,  Chen Weihang, danleifeng, feng_shuai, houj04, jiangcheng, JZ-LIANG, Kaipeng Deng, lidanqing, LielinJiang, Lijunhui, lilong12, liuyuhui, liym27, Pei Yang, Peihan, Qi Li, Ren Wei (任卫), Roc, Shang Zhizhou, ShenLiang, Shibo Tao, TeslaZhao, tianshuo78520a, TTerror, wangguanzhong, Wangzheee, wawltor, WeiXin, wenbin, Wenyu, whs, Wilber, wuhuanzhou, Zhang Ting, zhiboniu, Zhou Wei, zhoujun, 李季, 王明冬


# 2.1.0 Release Note

## Highlights

The PaddlePaddle Framework V2.1.0 has the following important updates:

- Environment Adaptation: Add the support for Python 3.9, CUDA 11.2; Provide the support for [ROCm platform](https://rocmdocs.amd.com/en/latest/) (experimental); Provide the support for [Ascend AI processor](https://e.huawei.com/cn/products/cloud-computing-dc/atlas/ascend-910) (experimental); Add the number of models that can run on [Baidu Kunlun chip](https://cloud.baidu.com/product/kunlun.html) . For details, please see: [Getting Started](https://www.paddlepaddle.org.cn/install/quick).

- Distributed training: besides [multidimensional hybrid parallelism](https://mp.weixin.qq.com/s/BblzcVn0NQ-QIhywvmoOuA) in static graph mode,  implementation in dynamic graph  is added.

- Framework function: Complete a number of enhancements and performance optimizations, in particular, including the following important new functions:
  
  - Customized OP: Provide a new solution for customizing operators outside the framework, simplifying the process of writing custom operators and deploying training inference. For details see: [Customizing External Operators](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/07_new_op/new_custom_op_cn.html).
  - Inplace Operation: Add the inplace operation to reduce the memory consumption and improve performance, including View strategy, and 12 inplace APIs.
  - High-level API related: Add the high-level APIs to support mixed precision training; add `paddle.hub` to view, share, and load models.
  - Automatic mixed precision training optimization: Optimized the computational performance of multiple OPs in mixed precision training such as slice, where, range, etc., and improved the acceleration effect on MaskRCNN, ERNIE and other models.
  - oneDNN BF16 training: Enabled AMP (AutoMixedPrecision) pure_BF16 mode. Enabled BF16 SGD and initializers for less memory consumption. Enabled most of FWD & BWD BF16 ops for BF16 word2vec training.

For the latest updates to the official model libraries and suites of PaddlePaddle, please see: [Paddle projects notes along with PaddlePaddle2.1](https://github.com/PaddlePaddle/Paddle/wiki/Paddle-projects-notes-along-with-PaddlePaddle2.1).

## Backwards Incompatible Changes

- The PaddlePaddle Framework 2.1 drops the support for python2 and python3.5. It is recommended that you upgrade your python to version 3.8 before using the PaddlePaddle. PaddlePaddle Framework 2.1 no longer provides the support for CUDA9 pre-built package. It is recommended that you upgrade the CUDA version before using the PaddlePaddle.
- The optimization of API visibility makes it impossible to import private APIs located in the deeply nested namespaces that are considered as implementation details by using `from deeply_nested_namespace import *`. It is recommended that you use the PaddlePaddle by following the instructions in the [API Documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html) on the PaddlePaddle website. Specifically, the following actions are no longer allowed in the PaddlePaddle Framework 2.1.

```python
# will import nothing from the deeply nested namespaces
from paddle.nn.layer.loss import *
from paddle.nn.layer.conv import *
```

- `Tensor.grad` Incompatible upgrade. The type of return value is changed from `numpy` to `Tensor`. ([#32142](https://github.com/PaddlePaddle/Paddle/pull/32142)) 


<table>
<tr>
<th>
2.0
</th>
<th>
2.1
</th>
</tr>

<tr>
<td>

```python
>>> import paddle
>>> x = paddle.to_tensor(5., stop_gradient=False)
>>> y = paddle.pow(x, 4.0)
>>> y.backward()
>>> type(x.grad)
< class ‘numpy.ndarray’ >
```
</td>

<td>

```python
>>> import paddle
>>> x = paddle.to_tensor(5., stop_gradient=False)
>>> y = paddle.pow(x, 4.0)
>>> y.backward()
>>> type(x.grad)
< class ‘paddle.Tensor’ >
```
</td>
</tr>
</table>


- `paddle.jit.TraceLayer.save_inference_model` Interface incompatibility upgrade. Changed the original first parameter dirname to path, the name symbol is more generic and unified with interfaces such as paddle.save and load, indicating that the user specifies the prefix for saving the model path. ([#31989](https://github.com/PaddlePaddle/Paddle/pull/31989)) 

<table>
<tr>
<th>
2.0
</th>
<th>
2.1
</th>
</tr>

<tr>
<td>

```python
>>> import os
>>> import paddle
>>> from paddle.vision.models import resnet18

>>> model = resnet18()
>>> x = paddle.rand([1, 3, 224, 224])
>>> _, static_layer = paddle.jit.TracedLayer.trace(model, inputs=[x])
>>> save_path = './save_infer_model'
>>> static_layer.save_inference_model(dirname=save_path)

>>> print(os.path.isdir(save_path))
>>> print(len(os.listdir(save_path)))
True
103
```
</td>

<td>

```python
>>> import os
>>> import paddle
>>> from paddle.vision.models import resnet18

>>> model = resnet18()
>>> x = paddle.rand([1, 3, 224, 224])
>>> _, static_layer = paddle.jit.TracedLayer.trace(model, inputs=[x])
>>> save_path = 'save_infer_model'
>>> static_layer.save_inference_model(path=save_path)

>>> print(os.path.isdir(save_path))
>>> print([name for name in os.listdir('./') if name.startswith(save_path)])
False
['save_infer_model.pdmodel', 'save_infer_model.pdiparams']
```
</td>
</tr>
</table>


- `paddle.io.DataLoader`return format incompatibility upgrade when user-define dataset only contains single field。If user-define dataset only contains single field and output data with code like `return image` or `yield image`，output data format in Paddle 2.0 is `[image_tensor]`，and output data format in Paddle 2.1 is `image_tensor` to keep data structure same with input.

<table>
<tr>
<th>
2.0
</th>
<th>
2.1
</th>
</tr>

<tr>
<td>

```python
>>> import numpy as np
>>> import paddle
>>> from paddle.io import DataLoader, Dataset
>>> 
>>> class RandomDataset(Dataset):
>>>     def __getitem__(self, idx):
>>>         return np.random.random((2, 3)).astype('float32')
>>> 
>>>     def __len__(self):
>>>         return 10
>>> 
>>> dataset = RandomDataset()
>>> loader = DataLoader(dataset, batch_size=1)
>>> data = next(loader())
>>> print(data)
[Tensor(shape=[1, 2, 3], dtype=float32, place=CUDAPinnedPlace, stop_gradient=True,
       [[[0.73782003, 0.62605530, 0.32727283],
         [0.37154925, 0.63570684, 0.53859973]]])]
```
</td>

<td>

```python
>>> import numpy as np
>>> import paddle
>>> from paddle.io import DataLoader, Dataset
>>> 
>>> class RandomDataset(Dataset):
>>>     def __getitem__(self, idx):
>>>         return np.random.random((2, 3)).astype('float32')
>>> 
>>>     def __len__(self):
>>>         return 10
>>> 
>>> dataset = RandomDataset()
>>> loader = DataLoader(dataset, batch_size=1)
>>> data = next(loader())
>>> print(data)
Tensor(shape=[1, 2, 3], dtype=float32, place=CUDAPinnedPlace, stop_gradient=True,
       [[[0.73782003, 0.62605530, 0.32727283],
         [0.37154925, 0.63570684, 0.53859973]]])
```
</td>
</tr>
</table>


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

<table>
<tr>
<th>
2.0
</th>
<th>
2.1
</th>
</tr>

<tr>
<td>

```python
>>> from paddle.vision.models import resnet18
>>> model = resnet18()
>>> 
>>> print(len(model.sublayers(include_sublayers=True)))
>>> print(len(model.sublayers(include_sublayers=False)))
67
0
```
</td>

<td>

```python
>>> from paddle.vision.models import resnet18
>>> model = resnet18()
>>> 
>>> print(len(model.sublayers(include_self=True)))
>>> print(len(model.sublayers(include_self=False)))
68
67
```
</td>
</tr>
</table>


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
- Added `paddle.distributed.send, paddle.distributed.recv, paddle.distributed.new_group, paddle.distributed.wait` to improve the distributed communication API. ([#32504](https://github.com/PaddlePaddle/Paddle/pull/32504),  [#31682](https://github.com/PaddlePaddle/Paddle/pull/31682))
- Support to initialize the  `sync_parameters_buffer`in the distributed dynamic graph, which solved the issue that the buffer of the dynamic graph is not globally initialized. ([#31625](https://github.com/PaddlePaddle/Paddle/pull/31625))
- Pipeline Parallelism supports 1F1B scheduling method to optimize the memory usage of GPU. Theoretically, it is constant([#31786](https://github.com/PaddlePaddle/Paddle/pull/31786)).
- \[Hybrid Parallel] Sharding strategy optimization: support Gradients aggregation, reducing the amount of parameter communication, and improving the speed of training; Could be used flexibly with other parallelism strategies. ([#31884](https://github.com/PaddlePaddle/Paddle/pull/31884) [#32486](https://github.com/PaddlePaddle/Paddle/pull/32486) [#32485](https://github.com/PaddlePaddle/Paddle/pull/32485) [#31996](https://github.com/PaddlePaddle/Paddle/pull/31996) [#31939](https://github.com/PaddlePaddle/Paddle/pull/31939) [#31796](https://github.com/PaddlePaddle/Paddle/pull/31796))
- \[Hybrid Parallel] Added optimizer state offload in the Sharding strategy, to reduce the memory usage of GPU. ([#32134](https://github.com/PaddlePaddle/Paddle/pull/32134))
- \[Hybrid Parallel] Support the persistence of the broadcast ID’s socket service, reduced the conflicts of ports in the hybrid parallelism. ([#31589](https://github.com/PaddlePaddle/Paddle/pull/31589))
- \[Parameter Server] Optimize the output and printing of LOG, and remove invalid logs.
- \[Parameter Server] Optimize the sparse parameter storage structure, with large memory reduction for small dimensions (below 64).
- \[Parameter Server] Fix the bug of access policy taking effect in the distributed prediction.
- HeterPs supports multiple machines. ([#31102](https://github.com/PaddlePaddle/Paddle/pull/31102))

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

- Add the support for Hygon chips:  PaddlePaddle, based on ROCM version 4.0.1, can train and infer models on Hygon CPU and DCU. A total of 36 models of 7 categories of image classification, target detection, image segmentation, natural language processing, recommendation systems, video classification and speech synthesis have been validated. ([#29342](https://github.com/PaddlePaddle/Paddle/pull/29342), [#30758](https://github.com/PaddlePaddle/Paddle/pull/30758), [#30639](https://github.com/PaddlePaddle/Paddle/pull/30639), [#31009](https://github.com/PaddlePaddle/Paddle/pull/31009), [#31077](https://github.com/PaddlePaddle/Paddle/pull/31077), and more)
- Add the support of Ascend chips: support for single hosting, multiple accelerators training on Ascend NPUs.  ([#31957](https://github.com/PaddlePaddle/Paddle/pull/31957), [#32381](https://github.com/PaddlePaddle/Paddle/pull/32381),  [#32197](https://github.com/PaddlePaddle/Paddle/pull/32197),  and more)
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