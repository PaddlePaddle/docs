# C++ OP 开发注意事项

## Paddle基于Yaml配置的算子代码自动生成
Paddle支持动态图和静态图两种模式，在Yaml配置文件中完成算子基本属性的定义后，需要进行解析并分别生成动态图和静态图所对应的算子代码逻辑，从而将算子接入框架的执行体系。基于Yaml配置的算子代码自动生成示意图：
![code_gen_by_yaml](./code_gen_by_yaml.png)

- 其中Yaml配置文件为前向：`python/paddle/utils/code_gen/api.yaml` 和反向：`python/paddle/utils/code_gen/backward.yaml`。
- 动态图中自动生成的代码包括从Python API到计算Kernel间的各层调用接口实现，从底层往上分别为：
  - C++ API：一套与Python API参数对齐的C++接口（只做逻辑计算，不支持自动微分），内部封装了底层kernel的选择和调用等逻辑，供上层灵活使用。
    - 注：前向算子生成C++ API头文件和实现代码分别为`paddle/phi/api/include/api.h`和`paddle/phi/api/lib/api.cc`，反向算子生成的头文件和实现代码分别为`paddle/phi/api/backward/backward_api.h`,`paddle/phi/api/lib/backward_api.cc`。
  - 动态图前向函数与反向节点（Autograd API）：在C++ API的基础上进行了封装，组成一个提供自动微分功能的C++函数接口。
    - 注：生成的相关代码在`paddle/fluid/eager/api/generated/eager_generated`目录下
  - Python-C 接口：将支持自动微分功能的C++的函数接口（Autograd API）暴露到Python层供Python API调用。
    - 注：生成的Python-C 接口代码在`paddle/fluid/pybind/eager_final_state_op_function_impl.h`中
- 静态图的执行流程与动态图不同，所以生成的代码也与动态图有较大差异。静态图由于是先组网后计算，Python API主要负责组网，算子的调度和kernel计算由静态图执行器来完成，因此自动生成的代码是将配置文件中的算子信息注册到框架内供执行器调度，主要包括[OpMaker](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/op_proto_maker.h)（静态图中定义算子的输入、输出以及属性等信息）和`REGISTER_OPERATOR`(将算子名称以及OpMaker等信息进行注册)等静态图算子注册组件，具体的代码逻辑可参考`paddle/fluid/operators/generated_op.cc`

**注意：由于代码自动生成在编译时进行，所以查看上述生成代码需要先完成[框架的编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/fromsource.html)。**

更多内容请参考: [C++ OP开发](new_cpp_op_cn.html)

## 写Op注意事项
### 1.Op兼容性问题
对Op的修改需要考虑兼容性问题，要保证Op修改之后，之前的模型都能够正常加载及运行，即新版本的Paddle预测库能成功加载运行旧版本训练的模型。<font color="#FF0000">**所以，需要保证Op当前的所有输入输出参数不能被修改（文档除外）或删除，可以新增参数，但是新增的Tensor类型变量需要设置为optional，非Tensor变量需要设置默认值。更多详细内容请参考[OP修改规范：Input/Output/Attribute只能做兼容修改](https://github.com/PaddlePaddle/Paddle/wiki/OP-Input-Output-Attribute-Compatibility-Modification)**</font> 。

### 2.显存优化

#### 2.1 为可原位计算的Op注册inplace
有些Op的计算逻辑中，输出可以复用输入的显存空间，也可称为原位计算。例如[reshape](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/reshape_kernel.cc)中，输出`out`可以复用输入`x`的显存空间，因为该Op的计算逻辑不会改变`x`的实际数据，只是修改它的shape，输出和输入复用同一块显存空间不影响结果。对于这类OP，可以注册`inlace`，从而让框架在运行时自动地进行显存优化。

注册方式为在算子的Yaml配置中添加`inplace`配置项，格式如：`(x -> out)`，详见[Yaml配置规则](new_cpp_op_cn.html#yaml)。示例：

```yaml
- api : reshape
  args : (Tensor x, IntArray shape)
  output : Tensor(out)
  ...
  inplace : (x -> out)
```

#### 2.2 减少OP中的无关变量
通常反向Op会依赖于前向Op的某些输入、输出Tensor，以供反向Op计算使用。但有些情况下，反向Op不需要前向Op的所有输入和输出；有些情况下，反向Op只需要前向Op的部分输入和输出；有些情况下，反向Op只需要使用前向Op中输入和输出变量的Shape和LoD信息。若Op开发者在注册反向Op时，将不必要的前向Op输入和输出作为反向Op的输入，会导致这部分显存无法被框架现有的显存优化策略优化，从而导致模型显存占用过高。

所以在定义反向Op时需要注意以下几点：

- 如果反向不需要前向的某些输入或输出参数，则无需在args中设置。
- 如果有些反向Op需要依赖前向Op的输入或输出变量的的Shape或LoD，但不依赖于变量中Tensor的内存Buffer数据，且不能根据其他变量推断出该Shape和LoD，则可以通过`no_need_buffer`对该变量进行配置，详见[Yaml配置规则](new_cpp_op_cn.html#yaml)。示例：
```yaml
- backward_api : trace_grad
  forward : trace (Tensor x, int offset, int axis1, int axis2) -> Tensor(out)
  args : (Tensor x, Tensor out_grad, int offset, int axis1, int axis2)
  output : Tensor(x_grad)
  ...
  no_need_buffer : x
```

### 3.ShareDataWith的调用
ShareDataWith的功能是使两个Tensor共享底层buffer，在调用这个操作的时候需要特别注意，在Op内部不能将ShareDataWith作用在Op的输出上，即Op输出的Tensor必须是Malloc出来的。

### 4.稀疏梯度参数更新方法
目前稀疏梯度在做更新的时候会先对梯度做merge，即对相同参数的梯度做累加，然后做参数以及附加参数（如velocity）的更新。

### 5.混合设备调用
由于GPU是异步执行的，当CPU调用返回之后，GPU端可能还没有真正的执行，所以如果在Op中创建了GPU运行时需要用到的临时变量，当GPU开始运行的时候，该临时变量可能在CPU端已经被释放，这样可能会导致GPU计算出错。

关于GPU中的一些同步和异步操作：
```
The following device operations are asynchronous with respect to the host:
    Kernel launches;
    Memory copies within a single device's memory;
    Memory copies from host to device of a memory block of 64 KB or less;
    Memory copies performed by functions that are suffixed with Async;
    Memory set function calls.
```

关于cudaMemCpy和cudaMemCpyAsync注意事项：

- 如果数据传输是从GPU端到非页锁定的CPU端，数据传输将是同步，即使调用的是异步拷贝操作。
- 如果数据传输是从CPU端到CPU端，数据传输将是同步的，即使调用的是异步拷贝操作。

更多内容可参考：[Asynchronous Concurrent Execution](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)，[API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior)

### 6. LoD 在 Op 内部的传导规范

[LoD](https://github.com/PaddlePaddle/FluidDoc/blob/develop/doc/fluid/design/concepts/lod_tensor.md) 是 Paddle 框架用来表示变长序列数据的属性，除了仅支持输入是 padding  data 的 Op 外，所有 Op 的实现都要考虑 LoD 的传导问题。

根据 OP 的计算过程中是否用到 LoD，我们可以将涉及到 LoD 传导问题的 OP 分为两类: LoD-Transparent 与 LoD-Based。

<table>
<thead>
<tr>
<th>类型</th>
<th>特点</th>
<th>示例</th>
</tr>
</thead>
<tbody>
<tr>
<td>LoD-Transparent </td>
<td>计算过程不依赖 LoD，输入是否有 LoD 不会影响计算的结果，通常是 position-wise 的计算 </td>
<td>conv2d_op、batch_norm_op、dropout_op 等 </td>
</tr>
<tr>
<td>LoD-Based </td>
<td>计算以序列为单位， 计算过程依赖 LoD </td>
<td> lstm_op、gru_op、sequence_ops 等 </td>
</tr>
</tbody>
</table>

这两类 OP 的 LoD 传导需要考虑前向和反向两个过程。

#### 前向传导

在前向传导过程，与输入的 LoD 相比较，Op 输出的 LoD 可能出现不变、改变和消失这三种情况：

  - 不变：适用于所有的 LoD-Transparent OP 与部分的 LoD-Based OP。可以在`InferMeta` 中调用 `ShareLoD()` 直接将输入 Var 的 LoD 共享给输出 Var, 可参考 [lstm_op](https://github.com/PaddlePaddle/Paddle/blob/a88a1faa48a42a8c3737deb0f05da968d200a7d3/paddle/fluid/operators/lstm_op.cc#L92); 如果有多个输入且都可能存在 LoD 的情况，通常默认共享第一个输入, 例如 [elementwise_ops forward](https://github.com/PaddlePaddle/Paddle/blob/5d6a1fcf16bcb48d2e66306b27d9994d9b07433c/paddle/fluid/operators/elementwise/elementwise_op.h#L69)；

  - 改变：适用于部分 LoD-Based OP。在实现 OpKernel 时需考虑输出 LoD 的正确计算，真实的 LoD 在前向计算结束后才能确定，此时仍需要在`InferMeta` 中调用 `ShareLoD()`，以确保CompileTime 时对 LoD Level 做了正确的传导，可参考 [sequence_expand_op](https://github.com/PaddlePaddle/Paddle/blob/565d30950138b9f831caa33904d9016cf53c6c2e/paddle/fluid/operators/sequence_ops/sequence_expand_op.cc)；

  - 消失：适用于输出不再是序列数据的 LoD-Based OP。此时不用再考虑前向的 LoD 传导问题，可参考 [sequence_pool_op](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/sequence_ops/sequence_pool_op.cc)；

其它重要的注意事项：

  - 实现 LoD-Based OP 时，需要处理好 LoD 传导的边界情况，例如对长度为零的输入的支持，并完善相应的单测，单测 case 覆盖空序列出现在 batch 开头、中间和末尾等位置的情况，可参考 [test_lstm_op.py](https://github.com/PaddlePaddle/Paddle/blob/4292bd8687ababc7737cffbddc0d38ead2138c00/python/paddle/fluid/tests/unittests/test_lstm_op.py#L203-L216)

  - 对 LoD Level 有明确要求的 OP，推荐的做法是在 `InferMeta` 中即完成 LoD Level的检查，例如 [sequence_pad_op](https://github.com/PaddlePaddle/Paddle/blob/4292bd8687ababc7737cffbddc0d38ead2138c00/paddle/fluid/operators/sequence_ops/sequence_pad_op.cc#L79)。


#### 反向传导

通常来讲，OP 的某个输入 Var 所对应的梯度 GradVar 的 LoD 应该与 Var 自身相同，所以应直接将 Var 的 LoD 共享给 GradVar，可以参考 [elementwise ops 的 backward](https://github.com/PaddlePaddle/Paddle/blob/a88a1faa48a42a8c3737deb0f05da968d200a7d3/paddle/fluid/operators/elementwise/elementwise_op.h#L189-L196)


## Op性能优化
### 1.第三方库的选择
在写Op过程中优先使用高性能（如cudnn、mkldnn、mklml、eigen等）中提供的操作，但是一定要做benchmark，有些库中的操作在深度学习任务中可能会比较慢。因为高性能库（如eigen等）中提供的操作为了更为通用，在性能方面可能并不是很好，通常深度学习模型中数据量较小，所以有些情况下可能高性能库中提供的某些操作速度较慢。比如Elementwise系列的所有Op（前向和反向），Elementwise操作在模型中调用的次数比较多，尤其是Elementwise_add，在很多操作之后都需要添加偏置项。在之前的实现中Elementwise_op直接调用Eigen库，由于Elementwise操作在很多情况下需要对数据做Broadcast，而实验发现Eigen库做Broadcast的速度比较慢，慢的原因在这个PR[#6229](https://github.com/PaddlePaddle/Paddle/pull/6229)中有描述。

### 2.Op性能优化
Op的计算速度与输入的数据量有关，对于某些Op可以根据输入数据的Shape和Op的属性参数来选择不同的计算方式。比如concat_op，当axis>=1时，在对多个tensor做拼接过程中需要对每个tensor做很多次拷贝，如果是在GPU上，需要调用cudaMemCopy。相对CPU而言，GPU属于外部设备，所以每次调用GPU的操作都会有一定的额外开销，并且当需要拷贝的次数较多时，这种开销就更为凸现。目前concat_op的实现会根据输入数据的Shape以及axis值来选择不同的调用方式，如果输入的tensor较多，且axis不等于0，则将多次拷贝操作转换成一个CUDA Kernel来完成；如果输入tensor较少，且axis等于0，使用直接进行拷贝。相关实验过程在该PR（[#8669](https://github.com/PaddlePaddle/Paddle/pull/8669)）中有介绍。

由于CUDA Kernel的调用有一定的额外开销，所以如果Op中出现多次调用CUDA Kernel，可能会影响Op的执行速度。比如之前的sequence_expand_op中包含很多CUDA Kernel，通常这些CUDA Kernel处理的数据量较小，所以频繁调用这样的Kernel会影响Op的计算速度，这种情况下最好将这些小的CUDA Kernel合并成一个。在优化sequence_expand_op过程（相关PR[#9289](https://github.com/PaddlePaddle/Paddle/pull/9289)）中就是采用这种思路，优化后的sequence_expand_op比之前的实现平均快出约1倍左右，相关实验细节在该PR（[#9289](https://github.com/PaddlePaddle/Paddle/pull/9289)）中有介绍。

减少CPU与GPU之间的拷贝和同步操作的次数。比如fetch操作，在每个迭代之后都会对模型参数进行更新并得到一个loss，并且数据从GPU端到没有页锁定的CPU端的拷贝是同步的，所以频繁的fetch多个参数会导致模型训练速度变慢。

## Op数值稳定性问题
### 1.有些Op存在数值稳定性问题
出现数值稳定性的主要原因程序在多次运行时，对浮点型数据施加操作的顺序可能不同，进而导致最终计算结果不同。而GPU是通过多线程并行计算的方式来加速计算的，所以很容易出现对浮点数施加操作的顺序不固定现象。

目前发现cudnn中的卷积操作、cudnn中的MaxPooling、CUDA中CudaAtomicXX、ParallelExecutor的Reduce模式下参数梯度的聚合等操作运行结果是非确定的。

为此Paddle中添加了一些FLAGS，比如使用FLAGS_cudnn_deterministic来强制cudnn使用确定性算法、FLAGS_cpu_deterministic强制CPU端的计算使用确定性方法。

## 其他
### 1.报错信息
Enforce提示信息不能为空，并且需要写明，因为报错信息可以更快更方便地分析出错误的原因。

### 2.Op的数学公式
如果Op有数学公式，一定要在代码中将数学公式写明，并在Python API的Doc中显示，因为用户在对比不同框架的计算结果时可能需要了解Paddle对Op是怎么实现的。

**注意：**在merge到develop分支之前一定进行公式预览。

### 3.Python端Op接口中参数的顺序
Python API中参数的顺序一般按照重要性来排，以fc为例：
```
def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       is_test=False,
       name=None)
```
