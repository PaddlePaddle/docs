
飞桨高可复用算子库 PHI (Paddle HIgh reusability operator library)，或者我们也成为函数式算子库，支持组合式算子功能复用、Primitive算子内核复用、插件式硬件加速库复用。针对飞桨框架原算子库存在的算子接口不清晰、算子复用成本较高、调用性能不够快的问题，我们重构了飞桨框架的算子库，设计了灵活、高效的函数式算子库 Phi，可以通过对函数式算子接口组合调用的方式实现新算子。新算子库提供了 200 余个跟 Python 开发接口保持一致的 C++ 运算类 API，以及近500个可供组合调用的前、反向函数式算子内核 Kernel，可大幅降低框架原生算子和自定义算子的开发成本。新算子库支持Primitive API方式开发算子内核，可支持不同硬件（比如GPU和XPU）的算子内核复用。新算子库支持以插件方式接入硬件（比如NPU）的加速库，实现低成本复用硬件加速库。

> 本文档撰写于phi架构基本成型之时（2022年2月），仅代表该时间点的基本设计形态，可能和最新形态有细微差别；此外，在2.3版本发布的phi算子库仍然处于初期形态，后续仍然需要持续建设并完善，设计上也有可能调整。

# 一、背景与目标

> 介绍设计并建设 phi 要解决的问题


最初 phi 项目的启动仅是为了优化飞桨动态图调度开销、并提升Kernel开发的复用能力而提出来的，但后续决定借此机会，建立能够同时在训练和推理场景（包括服务器端和移动端场景）中使用的“训推一体”算子库，长远上降低 paddle 生态中各基础设施开发及维护算子的成本，逐渐扩充了项目的目标范围，目前 phi 已经承载了多维度的意义。

> 关于算子库的命名，开发过程中有过迭代：初期算子库目录名为 pten ，意为paddle Tensor运算库 (Paddle Tensor Operation Library)，因此一些历史 PR 以PTen为前缀，后期认为该名称表述范围不够准确，因此更名为 phi

## 1.1 背景问题

具体地，phi 算子库项目，承载着解决 Paddle 以下问题的期望：

### 1.1.1 Op&OpKernel之间可复用性差，冗余代码较多

2.3版本之前，Paddle中的Operator（后续简称Op）之间的可复用性比较差，仅在少数的反向Op中，通过在GradOpMaker实现中调用SetType复用了一些简单的运算，大部分本身可以复用已有Op实现的情况，代码都是copy重写的。

可复用性差的根本原因还是原先Op体系设计导致的：

1. 当一个Op去复用另一个Op的`Opkernel::Compute`方法，都需要先构造一个`ExecutionContext`，复用上是比较繁琐的

	- 如果能直接调用一个函数形式的Kernel，就会方便很多

2. 由于额外的数据结构构造及独立Op调度引入了开销，从计算性能的角度考虑，复用Op不如直接把计算代码copy过来，导致我们逐渐抛弃了早期反向Op复用前向Op的原则，开始为每个反向Op单独实现Kernel

	- 只有Op之前复用的开销足够小，复用已有Op实现新Op才有可能被大范围推广

### 1.1.2 执行调度的简洁性与细粒度化

#### 1.1.2.1 动态图

Paddle 2.0发布之后，多次收到内外部用户反馈动态图在小模型CPU执行场景下与竞品在性能上有数倍的差距。

这个问题的主要原因是：Padddle动态图C++端的执行路径比较冗长，调度开销比较重，这和动态图早期设计兼容静态图，继承了静态图Op的许多对象构造过程有关

- 问题issue：https://github.com/PaddlePaddle/Paddle/issues/28774

因此，动态图需要升级为基于函数的调度架构，抛开原先复杂的Op体系，才能解决这个问题，这依赖于OpKernel改为函数式的写法。

#### 1.1.2.2 静态图 + IR

我们目前的静态图还不够“静态”，目前静态图仍然有许多运行时动态选择的逻辑，例如，运行时选择OpKernel，运行时判断是否要进行跨设备数据拷贝等等，但这些其实可以在静态图模型组网编译期间就确定下来，将执行过程确定为一系列OpKernel的执行，不再做动态的判断选择，从而进一步提升执行效率。

而这些依赖于OpKernel本身的细粒度化，将现有复杂的大OpKernel解耦成具体场景、具体设备的小Kernel，才能支持这样的调度。

### 1.1.3 自定义算子的易用性提升需求

2021年初上线的新自定义C++外部算子体系，在接口与函数编写的层面上，用法已经比较直观了，但是因为我们缺少基本运算的C++ API体系，事实上，在实现具体的自定义Op运算逻辑时，一些基础的加减乘除及矩阵运算都仍然需要重新实现一遍，不能复用Paddle已有的、经过优化的基础运算，因此一些复杂运算的外部开发成本仍然是比较高的。而要想复用Paddle内部的基础运算，有赖于的Op体系升级为函数式，并整理对应的C++ API体系才能解决。

### 1.1.4 共建训推一体算子库，降低推理算子维护成本

长久以来，由于paddle主框架和paddle-lite的算子是分开维护的，paddle新增的算子，lite需要的话，就要手动在lite中重新实现一遍，而且当主框架算子升级，lite又没有及时感知到，会直接导致推理模型在lite执行时出现bug，这维护成本是很高的，只有统一算子库，仅维护一份代码，才能长久解决这个问题。

因此，本次函数式算子库会由训练和推理共同建设，计算库整理完成后，作为独立的编译组件和底层基础设施（目前还没有独立拆分出来），能够同时服务于训练、预测以及Lite等执行体系。

### 1.1.5 推理新Runtime设计infrt的适配

推理设计了新的runtime infrt，预计要统一paddle-inference和paddle-lite的执行，将来需要直接调用本次共建的phi算子库中的算子，因此在设计时需要考虑对infrt的适配。

### 1.1.6 Op及Kernel参数规范化

Python 2.0 API项目规范了Paddle Python端API的参数列表，使其变得简洁、易用，但是限于当时的情况，Op层面的参数列表并没有规范化，因此会有不少早期开发的算子和Python API参数相差较多，例如conv op这种，Python API仅有7个参数，但C++ Op却有30+参数的分裂情况，而API和Op本质上是同一层概念，都是对一个运算的描述，参数应该是一致的。推理为了解决此问题，推动了算子定义增强项目，为部分不需要的参数添加了AsExtra以及AsQuant的声明，但并未从根本上解决问题，这也是phi算子库构建希望重点去解决的。

我们希望能做到，Python API -> Op(C++ API) -> Kernel API三层参数一致，使整体架构清晰，每一层复用也清晰，一套Python官方文档，基本能够满足三层API的共同参考需求，不再着重维护额外的文档体系，降低维护成本。

## 1.2 目标及范围

- 总体目标：飞桨核心框架复用同一函数式算子库，基础数据结构Tensor具备良好的可扩展性，从根本上做到训练推理协同一致、基础组件稳定可靠、增量开发体验良好。

- 目标范围：

  - phi算子库初期构建更关注Kernel“迁移”，人力因素，原Kernel逻辑迁移时暂不强制升级为“组合式”写法，前反向Kernel均如此
  - phi算子库初期提供的"组合式Kernel二次开发"能力面向后续增量的新算子使用，已有算子仍然保持其原先的编码实现，降低迁移成本
  - phi算子库初期提供的“新硬件扩展能力”仅在新硬件自身范围内提供，比如XPU已经实现了50个Kernel，后续其可以基于50个Kernel去组合新的Kernel，但这仅限于XPU范围内，其实现不和CPU、CUDA等实现通用
  - phi算子库项目重点关注“Kernel函数化&Op规范化”的工作，Kernel改为函数式，C++API与Op命名及参数列表在尽可能确保兼容性的前提下与逐渐规范化为与Python API一致


# 二、设计概览

## 2.1 命名及位置

飞桨高可复用算子库 (Paddle HIgh reusability operator library)，简称 PHI(phi)，phi代码目录在paddle目录下，和fluid平级，而不是放在fluid目录下，这样放置的原因是：phi是一个由fluid，lite，infrt等多种上层runtime共同调用的基础组件，后续会作为单独的编译的动态库存在，不适合作为fluid的子模块。

## 2.2 目录结构

### 2.2.1 目录结构设计需满足的需求

训练和推理对算子库目录的清晰度也有诸多诉求：

- 在目录设计上支持算子库的各种拆分编译需求，包括

	- 按运算设备拆分编译
		- 例如：仅编译cpu的，或者仅编译cuda的
	- 按训练和推理场景拆分编译
		- 例如：推理不编译反向相关kernel，也不编译带有Intermediate输出的前向kernel
	- 按移动端设备实际使用算子精准裁剪编译（目前尚未支持）
		- 例如：一个模型只用了add和mul，极致情况下应该能裁到仅剩2个kernel
- 长线上支持良好的kernel复用实现需求
	- 解释：kernel复用实现时，能否通过简单的include引入对应函数，不会因为目录过于复杂而找不到复用的kernel

- 长线上支持跨设备kernel的写法统一需求，并且直观易用，不引入不必要的模板参数
	- 解释：算子库下层还有Kernel Primitive API模块，其长线愿景是每个运算，只要一个kernel，能够适应多种设备，真正区分设备的代码，仅在Kernel Primitive API实现中；不希望未来的kernel在复用时从传入较复杂的模板参数，需要尽可能限制地简洁一些

- 易用性上，开发者能精准理解自己新增Kernel应该放到什么位置，无歧义
	- 解释：开发者新增一个API，不会困惑自己应该将对应kernel放在那个目录，也不会出现不同的人对于同一个kernel应该放在什么位置出现二义性的理解

- 不引入大量的重复目录设计
	- 解释：概念拆分是需要的，但也要有边界，避免在多个目录下有命名相同的子目录，容易混乱，比如不能cpu下面有eigen, funcs, math等，gpu下面也有。新算子库的目录设计以根据设备拆分为主，其他层次的目录拆分尽可能弱化，比如尽量不根据功能拆分，尽量不根据领域拆分等

- 不造成迁移时的文件数目膨胀
	- 解释：不能因为kernel设备拆分，导致kernel实现文件大规模增多

- 不引入层级过深的目录设计
	- 解释：目录层级不应过深，理解和维护成本都较高

- 不引入过高的迁移成本
	- 解释：迁移kernel时，不能要求对kernel本身做太多改动和拆分，否则迁移成本太高

### 2.2.2 具体目录设计

#### 2.2.2.1 一级目录

```
paddle/phi
./api (对外暴露的高层API及其实现)
	./include（对外暴露的高层API头文件）
	./lib（对外暴露API的实现）
./common (内外部均会使用到的基础数据结构)
./core (基础组件，比如基础Tensor相关接口，kernel注册接口，管理单元等)
./backends (各设备及后端的基础组件，下设cpu，gpu等后端目录)
./infermeta (shape、dtype、layout等meta信息的推导函数)
./kernels (各设备及后端的kernel实现)
./ops (各op的定义，后续采取自动生成的方式完成大部分工作，目前仅有兼容用的代码)
./tests (单元测试)
```

部分目录结构说明：

- `api`：API模块，面向外部用户
	- 直接使用类Python的C++ Tensor计算 API，和Python端形式高度一致
	- 该部分可能反向依赖框架的DeviceContextPool等实现，所以单独管理
	- 在该类API上，训练和预测也可能是不同的
- `common`：phi内部及phi api目录均要使用的数据结构，这些数据结构既不属于phi core，也不属于api目录
- `core`：phi内部会有一些自己需要的，公用的模块实现，比如基础DenseTensor、，kernel注册及管理模块
- `backends`：backends中组织后续需要为各个后端的新增的数据结构，比如CPUContext、GPUContext等
	- core中放置对于算子库来讲通用的基础数据结构，而特定后端的专用数据结构不放在core中，且依赖关系严格保证backends依赖core，但core不能依赖backends
	- 例1：Context如果有基类，则在core中，而继承的CPUContext在backends/cpu中，GPUContext在baackends/gpu中
	- 例2：TensorBase在core中，DenseTensor给多数设备使用，也在core中，如果有MKLDNNTensor的话，因为它只给mkldnn用，应该在backends/dnnl中
- `infermeta`: infermeta函数的整理位置，infermeta函数相当于infershape+inferdtype+inferlayout等
- `kernels`：各设备相关kernels
	- `cpu, gpu, ...`
- `ops`: ops中组织新形式的Op定义、以及兼容原有op的一些组件


#### 2.2.2.2 Kernels目录

```
paddle/phi/kernels
./ (放置设备无关的kernel声明和实现)
./cpu（仅放置cpu后端的kernel实现）
./gpu
./xpu
./dnnl
./gpudnn
./impl (考虑到现状，放置原先Kernel在CPU和GPU或其他设备一致的实现，便于复用)
./funcs（放置原fluid operators下一些支持多设备的functor和funcs）
./primitive（放置Kernel Primitive API的基础实现）
...
```

目录结构说明如下：

- kernels下主目录，放置设备无关的kernel.h和kernel.cc，原则上每个kernel一个.h和.cc
	- 例如一个kernel是使用Primitive api实现的，或者是复用其他基础kernel实现的，那么不论在什么设备上，应该都只有一种实现，所以它的声明和实现均直接放置到kernels目录下即可（这是将来的理想状态）
	- 目前我们大部分kernel都不具备跨设备实现统一的特征，但是kernel的输入参数返回值除了DeviceContext之外，应该是一致的，所以kernel参数声明头文件还放到主目录下（和原先的设计保持一致，DeviceContext和T作为模板参数），各设备的函数实现在相应的设备文件夹中
		- 注意，这里跨设备实现统一，并不是指一个kernel的CPU和GPU实现就算统一了，而是在所有设备的实现都一样，目前至少包括CPU，GPU，XPU，MKLDNN，GPUDNN等
	- 反向kernel如果不需要支持裁剪，可以做适当归并（但如果要为支持端侧训练留可能性，反向kernel可能也是裁剪的潜在目标）
- kernels下一级子目录，原则上按照backend分类按需新建，仅保留两个特殊的目录:
	- funcs：为了兼容原先fluid operators中functor和function设计保留的目录，放置支持多种后端的function和functor，还按照原先的一个头文件，多个.cc(u)的方式组织（这部分代码在将来可能被移除，因为会逐渐被Kernel Primirive API及Kernel间复用替代，这里不做过度设计）
		- 例1：一个公共函数XXXFunction在reduce CPU和reduce CUDA的kernel实现中都被调用，并且reduce CPU和reduce GPU的kernel实现是不一样的，那么这个XXXFunction应该在funcs目录中
	- primitive：Kernel Primitive API，多设备统一kernel实现的一些基础工具
	- impl：paddle目前的op kernel实现，有很多仍然是CPU和GPU复用同一份代码的，在大量的xx_op.h，这部分代码，不适合放在cpu或者gpu目录中，也不适合放在funcs目录中（会导致funcs目录中最终放置了相当一部分kernel实现，过于臃肿且混乱，funcs目录的定位是放置原先operators/math目录下那样的工具functor和function），也不适合放到kernels根目录下（并不是真正设备无关的实现，仅是cpu和gpu共用的实现），因此为了使这部分代码迁移时不需要做过多考虑，并且放置的位置也相对符合其实现性质，创建了impl这个目录
		- impl目录下，仅放置跨部分设备实现一致的kernel函数，均为头文件，命名均以xxx_kernel_impl.h为后缀
		- 例如：scale，fill_constant，fill_any_like这些kernel均属于此类情况
- kernel迁移过来之后，首先创建对应kenrel头文件直接放置到kernels的根目录中，各后端的kernel实现放在相应的设备文件夹中
	- 可参考原先op的归并程度，如matmul原先是单独的.h/.cc，那移过来之后保持，但activation相关的基本写在一个.h/.cc，移过来也仍然保持归并（后续有必要再进一步拆分）
	- 例1：原先cast op的Kernel在cast_op.h中，迁移过来之后在根目录创建cast_kernel.h，cast_kernel.cc/cu根据使用的后端放到对应的目录，即cast_kernel.cc放置到cpu中，cast_kernel.cu放置到gpu中
	- 例2：原先scale op的kernel使用eigen实现，CPU和GPU实现一致，迁移过来之后，公共实现应该在impl中的scale_kernel_impl.h中，公共头文件在kernels根目录下的scale_kernel.h中，scale_kernel.cc在cpu中，scale_kernel.cu在gpu中
- 迁移时，只有本kernel用到的辅助函数，一律和kernel实现放到同一个backend文件中，创建.h管理代码，不再单独在别处整理代码，除非这些辅助的函数实现是有多处使用的
	- 即使有多处调用，如果仍然限于同一设备，直接建头文件放到同一个目录下
- 反向kernel与前向kernel实现放置在不同的文件中，文件后缀采用``*_grad_kernel.*``，便于cmake分离编译
	- 不再为反向kernel单独创建目录，否则反向kernel目录下还要创建cpu/gpu等目录
	- 二阶导、三阶导的实现统一也放到grad kernel实现文件中

- 为什么目录名叫`gpu`而不是`cuda`和`hip`?
	- cuda和hip代码重复度非常高，统一实现维护成本较低


## 2.3 核心组件

### 2.3.1 公共基础数据结构

#### 2.3.1.1 Backend

```
/**
 * [ Why need Backend? ]
 *
 * Backend not only means place. Backend is a superset of place.
 *
 * Place cannot indicate the difference in calculation methods on the device,
 * but in order to make the boundary of the kernel clearer and the function
 * more specific, we need to distinguish the calculation method.
 *
 * Such as the kernel for CPU device, it can be a native CPU kernel,
 * or a kernel implemented by MKLDNN library.
 *
 * Note(chenweihang): HIP is not needed now, we can added it if needed
 * in the future
 */
enum class Backend : uint8_t {
  UNDEFINED = 0,

  // basic kernel backend
  CPU,

  // various acceleration devices' backends
  GPU,
  XPU,  // XPU currently does not exist at the same time as CUDA
  NPU,  // NPU currently does not exist at the same time as CUDA

  // the third library backend
  MKLDNN,
  GPUDNN,

  // end of backend types
  NUM_BACKENDS,

  /**
   * [ Why we need ALL in baisc kernel key member? ]
   *
   * For Tensor, ALL represents an illegal Backend, but for Kernel, some
   * kernels may be device-independent by nature, such as reshape; and when
   * and some kernels are also device-independent when implemented based on
   * primitive API.
   *
   * In this case, we need to provide a more concise registration method,
   * instead of registering the kernels for each device with almost
   * repetitive code, we need one registration covers all situations,
   * so if we provide the ALL field with Register the kernel in this statement.
   *
   * Of course, we have also considered solving this problem through different
   * named macros, for example, if we define
   *
   * PT_REGISTER_KERNEL_FOR_ALL_BACKEND
   *
   * Based on this design pattern, the dtype and layout also have the same
   * requirements, this cause we need to define a series of macros
   *
   * PT_REGISTER_KERNEL_FOR_ALL_DTYPE
   * PT_REGISTER_KERNEL_FOR_ALL_LAYOUT
   * PT_REGISTER_KERNEL_FOR_ALL_BACKEND_AND_LAYOUT
   * PT_REGISTER_KERNEL_FOR_ALL_BACKEND_AND_DTYPE
   * PT_REGISTER_KERNEL_FOR_ALL_LAYOUT_AND_DTYPE
   * PT_REGISTER_KERNEL_FOR_ALL_BACKEND_AND_LAYOUT_AND_DTYPE
   *
   * It makes the system of registering macros more complicated, we think
   * this is not a simple design, so we still adopt the design of providing
   * the ALL field.
   *
   * Note: ALL_BACKEND only used for Kernel registration and selection
   */
  ALL_BACKEND = UNDEFINED,
};
```

#### 2.3.1.2 DataLayout

```
// Note: Here the DataLayout is public api for external users, the prefix `k`
// maybe confuse users, so we use all uppercase names
enum class DataLayout {
  UNDEFINED = 0,
  // TODO(chenweihang): keep ANY for compatibility, remove it later
  ANY = UNDEFINED,
  NHWC,
  NCHW,
  MKLDNN,
  NUM_DATA_LAYOUTS,
  // See Note [ Why we need ALL in basic kernel key member? ]
  ALL_LAYOUT = UNDEFINED,
  // Note: Unify phi DataLayout and fluid::framework::DataLayout,
  // for compatible with fluid DataLayout, here need prefix `k`
  // Note: The original `kAnyLayout (enum value 2)` is a strange design.
  // `kAnyLayout` originally cannot represent any kind of Layout,
  // at the same time, it can also represent any Layout.
  // Strictly, it means "default" or "undefined" layout,
  // and should not be mixed with other meaningful layouts.
  kAnyLayout = ANY,
  kNHWC = NHWC,
  kNCHW = NCHW,
  kMKLDNN = MKLDNN,  // all layouts supported by MKLDNN internally
};
```

#### 2.3.1.3 DataType

```
enum class DataType {
  UNDEFINED = 0,
  BOOL,
  INT8,   // Char
  UINT8,  // BYte
  INT16,
  INT32,
  UINT32,
  INT64,
  UINT64,
  BFLOAT16,
  FLOAT16,
  UINT16,
  FLOAT32,
  FLOAT64,
  COMPLEX64,
  COMPLEX128,
  NUM_DATA_TYPES,
  // See Note [ Why we need ALL in baisc kernel key member? ]
  ALL_DTYPE = UNDEFINED,
};
```

- 这里什么不使用原先fluid的VarType？
    - 理由1：原先fluid的DataType和VarType是同级概念，设计是比较混乱的，例如LoDTensor和FLOAT32是同级概念，但这两者显然不是的，我们不希望继承原先有明显缺陷的设计
    - 理由2：和fluid解耦依赖，便于后续phi可以独立编译

#### 2.3.1.4 Scalar

Scalar (标量)用来统一表示具有不同基础数据类型(float, double, int, bool等)的变量。（目前也支持表示元素数量为1的Tensor标量，但后续可能会放弃该功能的支持）

以`ScaleKernel`为例，其中的`scale`参数可以传入int，float，double等普通数据类型。如果不使用`Scalar`来表示的话，需要为每种数据类型单独创建一个函数接口，这样会大大增加开发Kernel的代码量，因此`Scalar`主要应用在具有不同数据类型的同一参数上，可以避免该场景下需要编写多个重载函数的问题。

```
template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out);
```

#### 2.3.1.5 IntArray

IntArray 是一个整数类型数组，可以由`vector<int>`,`Tensor`以及`vector<Tensor>`进行构造，目前主要用来表示shape，index以及aixs等维度索引变量。

以FullKernel为例，其中的shape参数用来表示返回Tensor的维度信息（如[2，8，8]），在调用FullKernel时该项参数传入`vector<int>`,`Tensor`和`vector<Tensor>`类型的变量兼可完成调用。使用IntArray避免了每种shape类型单独编写一个重载函数的问题。

```
template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const IntArray& shape,
                const Scalar& val,
                DenseTensor* out);
```

### 2.3.2 Tensor体系

整体设计类图如下

![tensor-design.png](./images/tensor-design.png)


以下依次进行介绍。

#### 2.3.2.1 API Tensor接口

- 最上层是API级别的Tensor接口封装，里面包含两个指针成员，TensorBase和AbstractAutogradMeta。
	- 两个成员均使用了Interface设计，不会依赖于真实的Tensor和Autograd实现
	- AutogradMeta仅在动态图API级别的Tensor中有意义，在具体的kernel计算中，不会被使用到，所以将其放到最上层的Tensor接口中
	- 另外，这样设计也是为了方便数据共享，并且减少拷贝开销
		- 当一个Tensor赋值给另一个Tensor，或者Tensor作为函数返回值时，实际上只会拷贝指针，不会产生真实的数据拷贝

- 最上层C++ Tensor与Python端Tensor扮演类似的角色，在接口设计上尽可能与Python端保持一致
	- 包含基础的Tensor属性访问及数据访问方法
		- shape, place, dtype, data
	- 包含动态图Tensor需要的autograd方法
		- gradient, backward
	- 包含Tensor间的转换方法
		- cpu, gpu, xpu等
	- 包含tensor相关的计算方法（暂未添加）
		- `paddle.tensor` 模块下所有方法

- 编译解耦：

	- 这里带有的autograd信息，只是一个指针索引，默认为空
		- `std::unique_ptr<AbstractAutogradMeta> autograd_meta_ = nullptr;`
	- 而这里的AbstractAutogradMeta是一个抽象类接口，不会依赖autograd的任何模块，因此不会影响 phi 的独立编译，同时又兼顾了动态图Tensor需要持有反向信息的需求

- 这里的AutogradMeta仅在动态图场景中才会设置，不需要的场景，比如静态图内就仅仅是个空指针而已

Tensor设备的判断及转换

- Tensor的设备及类型判断

```
bool is_cpu() const;
bool is_gpu() const;
bool is_xpu() const;
bool is_dense_tensor() const;
bool is_selected_rows() const;
bool is_opencl() const; // 待添加
bool is_metal() const;  // 待添加
```

- Tensor间类型转换，通过与Python端一致的API实现（待添加）

```
Tensor cpu() const; // 转换为cpu tensor
Tensor gpu() const; // 转换为gpu tensor
Tensor xpu() const;
Tensor mkldnn() const;
```

- 这个转换的过程可能是cast，也可能是copy
	- 如果不需要进行数据拷贝，就是cast
	- 如果需要进行数据拷贝，就是copy
	- 转换通过函数式kernel去实现

- 在API场景中的使用
	- 用户在完整训练场景中，使用API的时候，最初读入的数据一般是从磁盘读入，先放入CPU，然后再转换到具体执行设备上，比如DataLoader

#### 2.3.2.2 TensorBase

- Tensor实现的接口类，接口中仅包含必要的纯虚Tensor方法，不包含有实际含义的成员，这里的方法在开发过程中也要严格控制

- 为什么要在这一层用抽象类设计？
	- 一方面是为了隔离Tensor API与Tensor具体实现，不产生过多依赖，如果将来Tensor API需要重新设计，或者说需要放弃掉autograd信息，只需要重新设计一个Tensor API即可，对于底层Tensor的实现几乎没有影响
	- 另一方面是为了给异构化的Tensor保留充足的扩展空间，框架API层仅需要一个Tensor数据结构即可，不需要再暴露多种数据结构设计，这里其实做了一个大范围定义，框架内所有数据结构均是Tensor
		- 对于内存布局基本一致，或者说Tensor描述基本一致的实现，可以基于一种DenseTensor的实现去继承
		- 如果是异构化程度高的Tensor，可以直接从Interface继承去实现新的Tensor分支，比如只有一个Object的Tensor，确保在Tensor扩展灵活性上不会出现瓶颈

#### 2.3.3.3 DenseTensor、SparseTensor

- 对应原fluid内的LoDTensor类，是Tensor的基类实现，Allocation就是现有Allocation，包含现有Tensor的基础成员
- SparseCsrTensor、SparseCooTensor为新设计的稀疏tensor类型，详见代码实现

> 为了兼容原先框架调度及算子，SelectedRows我们也迁移过来作为一种基础Tensor类型，后续如果能够被新的稀疏Tensor替代，长期会移除

#### 2.3.3.4 其他异构Tensor

- 如果现有Allocation的描述无法满足一些第三方库对于Tensor内存的描述需求，可以继承TensorBase之后，使用新的Allocation实现
- 而这种Tensor本质上没有脱离通用Tensor的范畴，只是访存方式有所区别，其他的TensorMeta信息，它仍然是需要的
- 可以自行定义特殊的TensorAllocation描述类，去构建自定义的Tensor，例如MetalTensor

```
template <typename AllocationType>
class SpatialTensor : public TensorBase {
 public:
  SpatialTensor(std::shared_ptr<AllocationType> allocation,
                std::unique_ptr<DenseTensorMeta> meta)
      : allocation_(std::move(allocation)),
        meta_(std::move(meta)) {}

 private:
  std::shared_ptr<AllocationType> allocation_;
  std::unique_ptr<TensorMeta> meta_;
};

template <typename AllocationType>
class MetalTensor : public SpatialTensor<AllocationType> {};

template <typename AllocationType>
class OpenCLTensor : public SpatialTensor<AllocationType> {};
```

- 通过这种方式，无论Tensor的需求如何特殊，均可以在对外API保持一致的前提下进行内部适配

其他高自由度Tensor继承：直接继承TensorBase

- TensorBase是抽象类，为具体Tensor的描述留了较大的空间，如果传统Tensor的描述无法满足需求，可以设计特异化的Tensor实现


### 2.3.3 C++ API

#### 2.3.3.1 C++ API形式

> 本节要点：
> 1. C++ API与Python 2.0 API对应，函数名、参数名、参数顺序、返回值均一致

经过调研，我们发现只有框架产品在设计时考虑了C++ API易用性层面的问题的。出于长期考虑，我们若想要吸引更多的开发者共建飞桨生态，提供规范易用的C++ API体系也是十分重要的。同时，Python 2.0 API项目为C++ API奠定了良好的参考基础，我们可以直接继承其成果。

因此，目前我们期望Tensor计算库的C++ API声明形式如下：

```
Tensor mean(const Tensor& x);

Tensor scale(const Tensor& x,
             const Scalar& scale,
             float bias,
             bool bias_after_scale);
```

说明如下：

- 尽可能与Python API属性保持一致，函数名，参数列表，返回值均保持一致，使用户在Python与C++的切换中，几乎没有新增的学习成本（如果必须不一致，可以增加新的C++ API，Python已有的运算类API与C++ API一一对应）

**这个新建的C++ API体系目前主要用于什么场景？**

1. 作为自定义算子开发时可调用的C++ API，提升易用性
	- 例如现在用户在自定义算子中初始化一个Tensor需要循环遍历Tensor数据并赋值，有API之后可以直接调用`paddle::ones`，`paddle::fill`这些API
2. 作为新动态图的基础调用单元
	- 新动态图会以API作为调度计算单元，不会再调用Op体系，以提升调度性能
3. 作为反向Op复用前向Op进行开发的基础
	- 现在反向op kernel需要单独实现，在API体系成型后，希望可以通过复用前向API完成反向Op实现

#### 2.3.3.2 C++ API自动生成

**为什么要自动生成C++ API？**

 - C++ API的实现代码在形式上相对固定，理论上可以采用自动生成的方式来实现
 - 使用代码自动生成可以有效降低C++ API的开发成本，且方便修改和维护

**如何自动生成C++ API？**

 C++ API的自动生成是通过解析Yaml配置文件来进行生成的，Yaml配置文件分为：

 - 前向API配置文件(`Python/paddle/utils/code_gen/api.yaml`，解析后生成代码文件为`paddle/phi/api/include/api.h`和`paddle/phi/api/lib/api.cc`)
 - 反向API配置文件(`Python/paddle/utils/code_gen/backward.yaml`，解析后生成的代码文件为`paddle/phi/api/backward/backward_api.h`和`paddle/phi/api/lib/backward_api.cc`)。

C++ API生成的关键在于Yaml文件的配置，以matmul为例，其前向和反向的配置文件如下：

```
# 前向API配置
- api : matmul
  args : (Tensor x, Tensor y, bool transpose_x=false, bool transpose_y=false)
  output : Tensor
  infer_meta :
    func : MatmulInferMeta
  kernel :
    func : matmul
  backward : matmul_grad

# 反向API配置
- backward_api : matmul_grad
  forward : matmul (Tensor x, Tensor y, bool transpose_x, bool transpose_y) -> Tensor(out)
  args : (Tensor x, Tensor y, Tensor out_grad, bool transpose_x=false, bool transpose_y=false)
  output : Tensor(x_grad), Tensor(y_grad)
  infer_meta :
    func : MatmulGradInferMeta
  kernel :
    func : matmul_grad
```

其中各项配置参数含义：

- api：函数名称，需与Phi Kernel注册的函数名相同
- args：函数参数，顺序和数据类型必须与Phi Kernel同名函数完全一致，Attributes类型必须排在Tensor类型之后。
- output：输出类型，如果有多个输出间用逗号(“,”) 分隔开。可以在类型后用"()"选择性标记每个输入的名字(如`Tensor(out)`)，如果没有标记则默认处理为out0, out1, …
- infer_meta：计算返回Tensor的维度与类型（详见InferMeta函数介绍）
 - func为调用的InferMeta函数，默认输入为args项的所有参数和output参数，其中的Tensor类型变量会自动替换为MetaTensor。
- kernel：API调用的具体Kernel函数
 - func：kernel函数的注册名（REGISTER使用的name，非函数名），默认输入为args项的所有参数和output参数
- backward：（可选）对应的反向函数名称，没有则生成纯前向API。

Yaml解析脚本将根据上述配置项自动生成对应的C++ API，生成的代码中会完成包括Kernel自动选择、Tensor转换、Data Transform、InferMeta以及Kernel调用等相关处理逻辑，具体可参考生成的`api.cc`内代码。

由于C++ API数量较多，且有着各种各样的形式与功能，为此在Yaml配置机制上也提供了很多更为灵活的配置项，如`invoke`等。

### 2.3.4 Kernel形式、注册及管理

#### 2.3.4.1 Kernel形式

> 本节要点：
> 1. Kernel函数形式要点：
> （1）数据类型T，与DeviceContext（简写为Context）作为模板参数；
> （2）Context作为Kernel第一个参数；
> （3）返回值Tensor以指针形式作为输入参数，Kernel本身返回值为void

这一层是具体的Kernel层，这一层实现的函数，会作为Kernel注册到框架中，供框架统一查找和调度。

目前我们期望这一层的形式如下，以`scale`为例：

```
template <typename T, typename Context>
void Scale(const Context& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  ...
}
```

说明如下：

- 不同设备的kernel要有不同的函数实现，函数名采用**驼峰式命名**，除了首字母大写之外，命名尽可能和API函数名保持一致，同一个计算的函数命名保持一致，通过不同文件或者目录管理不同设备的函数
- 一般有两个模板参数，T和Context（尽可能），用于运行时决定数据类型和设备类型
	- 按照我们目前的体系，绝大多数的Kernel都是按照**特化DeviceContext和数据类型**这种方式缩减代码的，这与原先OpKernel的形式一致性比较强
	- 形式要统一，将来如果Kernel层也作为细粒度API暴露的话，易用性有保障
- 函数输入参数规定：
	- 以具体的DeviceContext作为第一个输入参数，如CPUContext，CUDAContext，用于满足运行时需要特定上下文信息的需求，如多stream需要传stream进来
		- 暂不支持一个Kernel传入多个DeviceContext参数，目前认为这样的需求不太合理
	- 参数列表和API保持一致，如果有其他的特殊信息需要传入Kernel，通过Context传递
	- 随后是所有的输入Tensor与输入Attribute，均以const &方式传入，POD类型直接以值传入
	- 输入的Tensor是具体的Tensor类型，如DenseTensor或SelectedRows，不是对外接口API那个Tensor
	- 最后是函数的返回值Tensor，以指针形式传入
	- 为了满足灵活性，让kernel可以适配更多的场景，后续会允许声明灵活类型的输入、输出和参数，参考tfrt的Argument（输入）, Attribute,（属性） Return（输出）等模板，以适配非Tensor的输入输出，以及Tensor类的Attribute，让机制更加灵活
- 函数内部实现按需决定：
	- 短期：
		- 将现有OpKernel内实现，迁移到具体的设备Kernel内
		- 将存在设备公用的OpKernel实现抽离为函数，由多个设备Kernel共同调用
	- 长期：
		- 复杂Kernel直接调用基础Kernel完成计算，鼓励Kernel复用，简化代码

> FAQ：

>- 为什么需要使用模板参数？为什么不和torch一样，没有模板参数？
    - 运行时数据类型T和设备Device的选择是在kernel选择时必要的操作，各个框架都是一样的
    - torch在写法上避免了用模板实现kernel选择，但实际上采用了全局kernel map查找的选择方式，这种方式的开销是比较重的，一个kernel的执行过程中，可能存在多次kernel map的查找
    - 基本流程如下图：
        - ![图片](http://bos.bj.bce-internal.sdns.baidu.com/agroup-bos-bj/bj-2aafdb051eaea7120bdf9604eb738029dcd3162a)
    - 这种方式存在的性能问题已经被torch自身认识到，所以torch也在做算子库重构，但是积重难返，他们重构也并未对此问题从根本上解决，只是减少了一些redispatch的层数，我们不能一味模仿竞品自身都认为有问题的设计
- 为什么第一个参数需要是DeviceContext？为什么不能不传？
    - phi kernel要求是纯函数形式，即函数内使用的变量均通过参数传入，或者在函数内部创建，不允许在函数内部使用全局单例，为了适配多样的kernel需求，像DeviceContext这种存储上下文信息的参数是必要的
- 为什么需要两个模板参数？
    - 为了方便设备无关kernel的复用，假如我们要实现一个傅里叶变换fft kernel，假设这个kernel能够使用基础kernel组合得出，

#### 2.3.4.3 Kernel实现

> 本节要点：
> 1. Kernel专注表达数学算法，不掺杂调度逻辑
> 2. Kernel足够细粒度，边界清晰，没有可选参数，便于复用

现有Kernel因为Op参数过于复杂，引入了调度逻辑，例如

- 通过`use_cudnn`判断是否执行cudnn分支，在新的Tensor计算库中，使用cudnn计算是单独的Kernel

为了降低成本，Phi Kernel实现会尽可能继承原先的OpKernel实现，大部分Kernel的实现仅需要将原先OpKernel中取Input，Output的逻辑移除，并且修改一些关键方法即可，以sign为例：

原先sign OpKernel：

```
template <typename DeviceContext, typename T>
class SignKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& context) const {
    auto* out = context.Output<framework::Tensor>("Out");
    auto* in = context.Input<framework::Tensor>("X");
    out->mutable_data<T>(in->place());

    auto eigen_out = framework::EigenVector<T>::Flatten(*out);
    auto eigen_in = framework::EigenVector<T>::Flatten(*in);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    EigenSign<std::decay_t<decltype(place)>, T>::Eval(place, eigen_out,
                                                      eigen_in);
  }
};
```

迁移后的phi sign kernel：

```
template <typename T, typename Context>
void SignKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto eigen_out = phi::EigenVector<T>::Flatten(*out);
  auto eigen_x = phi::EigenVector<T>::Flatten(x);

  auto& dev = *dev_ctx.eigen_device();
  paddle::operators::EigenSign<std::decay_t<decltype(dev)>, T>::Eval(
      dev, eigen_out, eigen_x);
}
```

除了kernel形式从结构体变为函数式之外，还有两处主要变化：

1. 由于参数都是具体的输入，所以不需要再到context里取输入输出，相关代码移除
2. phi kernel中要求输出tensor的内存申请统一使用`ctx.Alloc`或者`ctx.HostAlloc`方法，不能再使用原先的`mutable_data`申请内存

> FAQ
> 1. 为什么mutable_data要替换成ctx.Alloc？
> 答：因为原先的mutable_data方法中调用的全局方法memory::AllocShared内部使用了全局单例进行内存分配，这不符合前面说过的纯函数设计原则，从业务需求上来讲，kernel里面如果使用单例确定显存分配的方式，在推理的多线程环境中，不能线程不能指定不同的存储分配方式。


#### 2.3.4.4 Kernel注册

> 本节要点：
> 1. Kernel将自身全部关键信息暴露给框架，记录其输入、输出和属性的信息，否则将导致框架调度与 Kernel 计算之间界限不清

现有 fluid Kernel 注册时仅记录了 Kernel 的 place，layout，dtype，输入输出等统一由 ExecutionContext管理，没有相应的信息记录，现在kernel要改成函数式，每一个函数的输入输出和属性都是明确的，我们希望在这里记录每一个输入输出的信息，也是为了兼容paddle-lite的调度。

同时，我们需要简化Kernel注册的写法，现有的写法都不够简洁：

1. fluid的Kernel注册写法，有不少冗余信息，以scale为例，可以看到每个kernel除了最后的data type，前面函数名和DeviceContext特化的信息都是冗余的

	```
	REGISTER_OP_CPU_KERNEL(
	    scale, ops::ScaleKernel<paddle::platform::CPUDeviceContext, float>,
	    ops::ScaleKernel<paddle::platform::CPUDeviceContext, double>,
	    ops::ScaleKernel<paddle::platform::CPUDeviceContext,
	                     paddle::platform::bfloat16>,
	    ops::ScaleKernel<paddle::platform::CPUDeviceContext, uint8_t>,
	    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int8_t>,
	    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int16_t>,
	    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int>,
	    ops::ScaleKernel<paddle::platform::CPUDeviceContext, int64_t>);
	```

2. Paddle-Lite的kernel注册写法，为每一个Kernel都声明了输入输出信息，但由于每个数据类型的kernel都是不同的，也会造成写法上的冗余，如下代码可以看到，除了data type，其他的信息也基本是冗余的

	```
	#ifdef LITE_BUILD_EXTRA
	using scale_int32_f =
	    paddle::lite::kernels::arm::ScaleCompute<int, PRECISION(kFloat)>;
	REGISTER_LITE_KERNEL(scale, kARM, kFloat, kNCHW, scale_int32_f, int32)
	    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
	    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
	    .Finalize();

	using scale_int64_f =
	    paddle::lite::kernels::arm::ScaleCompute<int64_t, PRECISION(kFloat)>;
	REGISTER_LITE_KERNEL(scale, kARM, kFloat, kNCHW, scale_int64_f, int64)
	    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
	    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
	    .Finalize();
	#endif  // LITE_BUILD_EXTRA

	#ifdef ENABLE_ARM_FP16
	using scale_float16 =
	    paddle::lite::kernels::arm::ScaleCompute<float16_t, PRECISION(kFP16)>;
	REGISTER_LITE_KERNEL(scale, kARM, kFP16, kNCHW, scale_float16, def)
	    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
	    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFP16))})
	    .Finalize();

	#endif  // ENABLE_ARM_FP16

	using scale_float =
	    paddle::lite::kernels::arm::ScaleCompute<float, PRECISION(kFloat)>;
	REGISTER_LITE_KERNEL(scale, kARM, kFloat, kNCHW, scale_float, def)
	    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
	    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kFloat))})
	    .Finalize();

	using scale_int32 =
	    paddle::lite::kernels::arm::ScaleCompute<int, PRECISION(kInt32)>;
	REGISTER_LITE_KERNEL(scale, kARM, kInt32, kNCHW, scale_int32, def)
	    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
	    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
	    .Finalize();

	using scale_int64 =
	    paddle::lite::kernels::arm::ScaleCompute<int64_t, PRECISION(kInt64)>;
	REGISTER_LITE_KERNEL(scale, kARM, kInt64, kNCHW, scale_int64, def)
	    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
	    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt64))})
	    .Finalize();
	```

因此，本次设计，不希望继续保持目前这种冗余的写法，希望kernel注册方法足够简洁，同时还能够灵活地满足Kernel输入输出信息配置的需求。

对于这个问题，关键点在于kernel需要指定自己的device，layout和dtype作为它自己的key信息，而大部分kernel输入输出Tensor的device，layout和dtype和kernel自身是一致的，对于这类kernel，我们可以按照kernel的信息自动生成填充每个输入输出的信息，不需要通过BindInput，BindOutput声明；我们只需要针对与kernel信息不一致的输入输出去配置特殊信息即可。

新实现的kernel注册形式如下：

```
PT_REGISTER_KERNEL("sign", CPU, NCHW, pt::Sign, float, double) {}

PT_REGISTER_KERNEL("mean", CPU, NCHW, pt::Mean, float, double) {}

PT_REGISTER_KERNEL("scale", CPU, NCHW, pt::Scale, float, double, bfloat16,
                   uint8_t, int8_t, int16_t, int, int64_t) {}

PT_REGISTER_KERNEL("scale.host", CPU, NCHW, pt::ScaleHost, float, double, bfloat16,
                   uint8_t, int8_t, int16_t, int, int64_t) {
   kernel->InputAt(1).SetBackend(pt::Backend::kCPU);
}
```

说明如下：

- 去除了之前注册方法中大量的冗余信息，可以一行代码完成8个数据类型的scale kernel注册，同时根据kernel信息默认记录每个输入输出的信息
- 对于有`ScaleTensor`这种动态attr输入的kernel，可以在函数体重配置具体参数的Backend，Layout和Dtype信息；没有此类需求的，函数体为空即可

此外，在`PT_REGISTER_KERNEL`宏内，通过模板推导，对Kernel函数的函数形式了归一化处理。

输入参数列表各异的kernel统一被归一化为如下形式，从而能够以统一的函数指针存储到下文中的Kernel数据结构中：

```
using KernelFn = void (*)(KernelContext* ctx);
```

通过在Kernel函数外包裹`PT_KERNEL`进行自动推导

```
#define PT_KERNEL(...) \
  ::pt::KernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute
```

此外，目前仅实现了基本的模板适配，后续我们会根据需求添加，以让在整体机制更加灵活，适用范围更广。

#### 2.3.4.4 Kernel管理

> 本节要点：
> 1. 介绍目前Kernel管理组件的设计

对于新形式Kernel的管理，目前设计类图如下：

![kernel-design.png](./images/kernel-design.png)

说明如下：

- `KernelFactory`作为管理Kernel的全局单例数据结构，和fluid的OpKernelMap类似，两级map，第一层根据name找到Kernel集合，第二层根据KernelKey找到具体的Kernel
- `KernelKey`和原先的OpKernelType类似，但将palce和library_type字段合二为一称之为Backend，因为原先的LibraryType是一个有局限的枚举类，原本就和place是强相关的，拆分反而增加了理解成本
- `Kernel`相比原先的OpKernel持有了更多信息，除了执行时的Function，还持有了具体参数的信息，即`KernelArgsDef`，对于Tensor类输入输出，保存了Tensor类型信息、Device，数据类型、数据布局，对于Attribute类输入输出，保存了类型信息


### 2.3.5 Kernel自动化编译及依赖分析

> 本节要点：
> 1. 介绍kernel的自动化编译设计
> 2. 介绍kernel的自动化依赖分析设计

原OpKernel迁移至phi之后，在编译上需要创建新的编译target，目前phi也设计了相应的自动化编译方式，使大家在迁移之后，尽可能不需要关注编译相关的内容。

#### 2.3.5.1 Kernel自动化编译

目前按照相应的规范迁移kernel之后，重新执行cmake，cmake会自动根据新增kernel的文件名，创建相应的编译对象，相关的逻辑在`paddle/phi/kernels/CMakeLists.txt`

```
set(COMMON_KERNEL_DEPS dense_tensor sparse_coo_tensor sparse_csr_tensor kernel_context kernel_factory arg_map_context convert_utils lod_utils)
set(COMMON_KERNEL_DEPS ${COMMON_KERNEL_DEPS} eigen_function blas math_function)
# remove this dep after removing fluid deps on tensor creation
set(COMMON_KERNEL_DEPS ${COMMON_KERNEL_DEPS} phi_api_utils)
set(COMMON_KERNEL_DEPS ${COMMON_KERNEL_DEPS} infermeta)

# auto build kernel targets by cmake
register_kernels(EXCLUDES math_kernel DEPS ${COMMON_KERNEL_DEPS})

set(MATH_KERNEL_DEPS ${COMMON_KERNEL_DEPS} cast_kernel copy_kernel phi_transpose_cpu)
if(WITH_GPU OR WITH_ROCM)
  set(MATH_KERNEL_DEPS ${MATH_KERNEL_DEPS} phi_transpose_gpu)
endif()
kernel_library(math_kernel DEPS ${MATH_KERNEL_DEPS})
```

1. 首先，定义kernel的公共依赖集合`COMMON_KERNEL_DEPS`，有较多kernel依赖的组件均可以放置到该集合中
2. 通过函数`register_kernels`，自动解析kernels目录下的`***_kernel.h`文件，自动创建对应的target
3. 如果某个kernel有自己独特的依赖，可以将其标记在`register_kernels`的EXCLUDES集合中，跳过对其的自动生成，后面再使用`kernel_library`函数，生成对应的kernel target，`kernel_library`也是根据文件名自动生成编译target的

具体`register_kernels`和`kernel_library`如果扫描文件并生成编译对象，可以参考`camke/phi.cmake`中的函数实现，此处不展开介绍了

#### 2.3.5.2 Kernel依赖自动化分析

phi kernel整体改为了函数式，本意就是让kernel之间可以更加方便地复用，但是复用kernel会引入kernel之间的编译依赖关系，比如A  Kernel调用了B Kernel，那么在编译上，A Kernel需要DEPS B Kernel，这样的编译依赖声明对于开发者来讲同样是非常繁琐的，因此我们也设计了对应的自动化解析方式，具体如下：

在编译A Kernel时，我们会分析A Kernel相关的`.h`和`.cc/cu`文件中include声明，如果A Kernel include了 B Kernel的头文件声明，我们会自动为A Kernel添加B Kernel target的依赖，例如：

dot_kernel.h有`#include "paddle/phi/kernels/empty_kernel.h"`，那么在编译时，dot_kernel会自动依赖empty_kernel，这一过程也是在`register_kernels`和`kernel_library`函数中实现的，可以参考`camke/phi.cmake`中的函数实现。

因此，开发时如果需要进行Kernel复用，正确include相应头文件即可。

> 注意：这里只有kernel间的复用是会自动解析的，如果某个kernel依赖了某个function或者functor，仍然是需要手动声明依赖的，phi的设计鼓励kernel之间的复用，因为kernel本身也成为function了，因此像之前那种调用function的方式长期来讲是基本可以被淘汰掉的，只需要尽可能将function实现为kernel即可

### 2.3.6 InferMeta(Shape)抽象整合

原先fluid Op的InferShape和OpKernel一样，存在重复开发的问题，因为不同Op的InferShape函数无法复用，因此即使不同Op的InferShape逻辑一样或者类似，也都是重写一遍，本次phi的重构也需要解决此问题。

我们将InferShape同样改写为函数式，支持不同的Op可以调用同一个InferShape函数，提升易用性，降低维护成本。

> FAQ：
> 1. 为什么要叫InferMeta，而不是继续叫InferShape？
> 答：InferMeta的Meta来源于DenseTensor中的meta成员，在phi中，一个op有两大组件，InferMeta和Kernel。这里InferMeta覆盖了InferShape的功能，但又不限于InferShape，除了对dims和lod的推断，InferMeta中也会承担dtype和layout的推断，这一点和原先是不一样的。

#### 2.3.6.1 InferMeta相关设计

首先InferMeta也为函数式，几个示例如下：

```
void UnchangedInferMeta(const MetaTensor& x, MetaTensor* out) {
  out->share_meta(x);
}

void CastInferMeta(const MetaTensor& x, DataType out_dtype, MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(out_dtype);
  out->set_layout(x.layout());
}

void CreateLikeInferMeta(const MetaTensor& x,
                         DataType dtype,
                         DataLayout layout,
                         MetaTensor* out) {
  out->set_dims(x.dims());
  out->set_dtype(dtype == DataType::UNDEFINED ? x.dtype() : dtype);
  out->set_layout(layout == DataLayout::UNDEFINED ? x.layout() : layout);
}

void ConcatInferMeta(const std::vector<MetaTensor>& x,
                     const Scalar& axis_scalar,
                     MetaTensor* out,
                     MetaConfig config = MetaConfig());
```

特征介绍如下：

1. 函数命名为`[FunctionDesc|OpName]InferMeta`
2. 函数形式与Kernel类似，函数参数依次为MetaTensor输入，Attribute，MetaTensor输出，返回值为空，原则上InferMeta函数与其对应Kernel函数的参数列表是一一对应的，差别仅为Tensor参数类型，InferMeta函数的Tensor参数为MetaTensor，Kernel函数的Tensor参数为DenseTensor，SparseTensor等
3. 对于一些需要区分编译期与执行期的InferMeta函数，在末尾添加MetaConfig参数，config中有is_runtime的flag成员，之所以用结构体，是为了便于后续扩展其他flag成员。

这里使用MetaTensor是为了屏蔽多种Tensor类型，以及兼容原先fluid的VarDesc及Variable，一个op对应一个InferMeta函数即可，如果不对类型进行屏蔽，本身InferMeta函数就会因为输入类型不同而重复开发多份。

其中MetaTensor的基础设计如下：

```
class MetaTensor {
 public:
  explicit MetaTensor(TensorBase* tensor) : tensor_(tensor) {}

  MetaTensor() = default;
  MetaTensor(const MetaTensor&) = default;
  MetaTensor(MetaTensor&&) = default;
  MetaTensor& operator=(const MetaTensor&) = delete;
  MetaTensor& operator=(MetaTensor&&) = delete;

  virtual ~MetaTensor() = default;

  virtual int64_t numel() const;
  virtual DDim dims() const;
  virtual DataType dtype() const;
  virtual DataLayout layout() const;
  virtual void set_dims(const DDim& dims);
  virtual void set_dtype(DataType dtype);
  virtual void set_layout(DataLayout layout);
  virtual void share_lod(const MetaTensor& meta_tensor);

 private:
  const LoD& lod() const;
  TensorBase* tensor_;
};
```

基类的MetaTensor中有一个TensorBase的指针成员，因此在phi中可以兼容DenseTensor，SelectedRows，SparseTensor等多种类型。

#### 2.3.6.2 InferMeta注册管理

为了支持InferMeta函数的统一调用，InferMeta函数也进行了统一的注册管理。

首先也需要类似前述Kernel形式归一化的`PT_KERTNEL`工具宏，命名为`PT_INFER_META`，并实现类似KernelContext的InferMetaContext（实现不展开了，仅放置部分片段，详见`phi/core/infermeta_utils.h`）

```
class InferMetaContext {
 public:
  InferMetaContext() = default;
 ...
};

#define PT_INFER_META(...) \
  ::phi::InferMetaFnImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Call

template <typename Fn, Fn fn>
struct InferMetaFnImpl;

template <typename Return, typename... Args, Return (*infer_meta_fn)(Args...)>
struct InferMetaFnImpl<Return (*)(Args...), infer_meta_fn> {
  static void Call(InferMetaContext* ctx) {
    InferMetaFnCallHelper<Args..., InferMetaTypeTag<int>>::template Call<0, 0, 0>(ctx);
  }

 private:
  template <typename... RemainingArgs>
  struct InferMetaFnCallHelper;

  ...
};
```

然后设计对应的单例类用来存储MetaFn

```
class MetaFnFactory {
 public:
  static MetaFnFactory& Instance();

  bool Contains(const std::string& kernel_name_prefix) const {
    return meta_fn_map_.count(kernel_name_prefix) > 0;
  }

  void Insert(std::string kernel_name_prefix, InferMetaFn infer_meta_fn) {
    PADDLE_ENFORCE_NE(
        Contains(kernel_name_prefix),
        true,
        phi::errors::AlreadyExists(
            "`%s`'s Series Kernel's InferMetaFn has been registered.",
            kernel_name_prefix));
    meta_fn_map_.insert(
        {std::move(kernel_name_prefix), std::move(infer_meta_fn)});
  }

  const InferMetaFn& Get(const std::string& kernel_name_prefix) const {
    auto it = meta_fn_map_.find(kernel_name_prefix);
    PADDLE_ENFORCE_NE(
        it,
        meta_fn_map_.end(),
        phi::errors::NotFound(
            "`%s`'s Series Kernel's InferMetaFn is not registered.",
            kernel_name_prefix));
    return it->second;
  }

 private:
  MetaFnFactory() = default;

  /**
   * [ Why use kernel name prefix? ]
   *
   * one op -> a matrix of kernels
   *
   * such as, scale op, it may correspond to the following kernels:
   *
   * - scale, scale_sr, scale_dnnl
   * - scale_raw, scale_raw_sr, scale_raw_dnnl
   *
   * All the kernels in each row correspond to the same infershape function,
   * the number of kernel arguments in the same row is the same, and only
   * the tensor types in the arguments are different.
   */
  paddle::flat_hash_map<std::string, InferMetaFn> meta_fn_map_;

  DISABLE_COPY_AND_ASSIGN(MetaFnFactory);
};
```

封装对应的注册宏，用于InferMeta的注册，注册写法示例如下：

```
PT_REGISTER_INFER_META_FN(sign, phi::UnchangedInferMeta);
```

对于InferMeta的注册，一般不需要开发者手写，我们通过yaml中api name和InferMeta的映射关系，自动生成对应的注册条目。

#### 2.3.6.3 InferMeta兼容fluid InferShape

在fluid中，继承MetaTensor实现CompatMetaTensor，重写对应的成员方法，以使InferMeta函数兼容VarDesc和Variable的输入，以dims为例，CompatMetaTensor的dims实现为：

```
class CompatMetaTensor : public phi::MetaTensor {
 public:
  CompatMetaTensor(InferShapeVarPtr var, bool is_runtime)
      : var_(std::move(var)), is_runtime_(is_runtime) {}

  CompatMetaTensor() = default;
  CompatMetaTensor(const CompatMetaTensor&) = default;
  CompatMetaTensor(CompatMetaTensor&&) = default;
  CompatMetaTensor& operator=(const CompatMetaTensor&) = delete;
  CompatMetaTensor& operator=(CompatMetaTensor&&) = delete;

  ...

  DDim dims() const override {
    if (is_runtime_) {
      auto* var = BOOST_GET_CONST(Variable*, var_);
      if (var->IsType<phi::DenseTensor>()) {
        return var->Get<phi::DenseTensor>().dims();
      } else if (var->IsType<phi::SelectedRows>()) {
        return var->Get<phi::SelectedRows>().dims();
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Currently, only can get dims from DenseTensor or SelectedRows."));
      }
    } else {
      auto* var = BOOST_GET_CONST(VarDesc*, var_);
      return make_ddim(var->GetShape());
    }
  }
  ...
};
```

然后，为了将函数式的InferMeta嫁接回fluid的Op体系上，需要将函数式的InferMeta归一化为functor形式。

通过前面介绍的PT_INFER_META宏归一化函数形式，然后将`PT_INFER_META(***InferMeta)`包装到一个functor中，functor中先将InferShapeContext转换为InferMetaContext，再调用相应InferMeta函数，通过一个宏统一管理代码

```
#define DELCARE_INFER_SHAPE_FUNCTOR(op_type, functor_name, fn)      \
  struct functor_name : public paddle::framework::InferShapeBase {  \
    void operator()(                                                \
        paddle::framework::InferShapeContext* ctx) const override { \
      auto infer_meta_context =                                     \
          paddle::framework::BuildInferMetaContext(ctx, #op_type);  \
      fn(&infer_meta_context);                                      \
    }                                                               \
  }
```

这其中的关键函数是`BuildInferMetaContext`，这个函数会从InferShapeContext中，将InferMeta函数需要的参数取出，统一放到InferMetaContext中并返回，InferMeta需要的参数列表通过ArgumentMapping函数获取（详细在2.4 动静态图执行兼容适配中介绍）。

然后将该functor在Op注册时维护到相应OpInfo中即可，同时删除原先Op的InferShape实现，示例如下

```
// 原先实现
class SignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "sign");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "sign");

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

namespace ops = paddle::operators;

REGISTER_OPERATOR(sign, ops::SignOp, ops::SignOpMaker<float>,
                  ops::SignGradMaker<paddle::framework::OpDesc>,
                  ops::SignGradMaker<paddle::imperative::OpBase>);

// 升级后实现
class SignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

DELCARE_INFER_SHAPE_FUNCTOR(
    sign, SignInferShapeFunctor, PT_INFER_META(phi::UnchangedInferMetaNew));
REGISTER_OPERATOR(sign, ops::SignOp, ops::SignOpMaker<float>,
                  ops::SignGradMaker<paddle::framework::OpDesc>,
                  ops::SignGradMaker<paddle::imperative::OpBase>,
                  SignInferShapeFunctor);

```

至此，实现原Op的InferShape函数迁移至phi InferMeta之后，可以重新注册回fluid中被调用，从而实现InferShape的函数化复用与全局统一。

## 2.4 动静态图执行兼容适配

> 本节要点：
> 1. 新形式Kernel如何在现有静态图和动态图体系中调用，难点在于解决多参数Op到少参数Kernel的匹配问题

### 2.4.1 ArgumentMapping体系设计

由于新形式Kernel参数列表与Python API对齐，和原先的OpMaker中注册的参数列表存在差异，导致新形式Kernel在原先fluid体系中调用时会很难匹配

例如conv2d op，它的OpMaker中注册了4个Input，1个Output，26个Attribute，而conv2d的Python API一共只有8个参数（不算name，3个Tensor输入，5个Attribute输入）

运行时，调用新Kernel之前，需要将Kernel需要的参数从OpMaker注册的参数中选出来，再传给新Kernel使用。

对于一些原先就编写规范的算子，它的OpMaker参数和Python api参数本就是对应的，这种标准的情况，不存在需要选参数的需求，对于这部分算子，根据OpProto中输入输出属性的注册顺序，跳过标记为Extra和Quant的成员，可以解决一部分Op和Kernel的参数匹配问题；然而对于一些不太规范，或者说是fluid时代遗留的算子，比如像conv，就需要这样的映射函数，且这个映射函数根据op不同，可能存在非常复杂的判断逻辑，因此现阶段没有办法可以自动化处理。

为此，目前设计了ArgumentMapping函数映射的体系，在phi/ops/compat目录下，实现相应的映射函数并注册，然后在phi kernel执行适配时，会调用对应的ArgumentMapping函数，得到phi kernel需要的参数，例如scale op的映射函数如下：

```
/**
 * Note [ Why does the ArgumentMapping function need to be so complicated? ]
 *
 * In order to meet the requirements of infrt, the function used to match Op
 * and Kernel parameters, need to be placed in phi as a compatible component,
 * and does not depend on fluid.
 *
 * Because infrt not only needs to dynamically call this argument mapping
 * function at runtime, but also needs to statically declare all possible
 * results of the function before running without any information.
 *
 * The infrt declare like:
 *
 * def PDKEL_Reshape_to_CPU : Pat<
 *     (PD_ReshapeOp $x, $shape_tensor， $shape_attr), // OpMaker arguements
 *     (PDKEL_ReshapeKernelAttr $x, fn($shape_attr)>;  // Kernel arguments
 * def PDKEL_Reshape_to_CPU : Pat<
 *     (PD_ReshapeOp $x, $shape_tensor， $shape_attr),
 *     (PDKEL_ReshapeKernelAttr $x, fn($shape_tensor)>;
 *
 * Therefore, we need to write out each result of the argument mapping function,
 * like `KernelSignature("full", {}, {"ShapeTensor", "value"}, {"Out"})`, it
 * cannot contains variable, only can contains const char* string.
 *
 * Infrt will parse all results before running for the generation of the above
 * static declare, which leads to some functions being written in a long way,
 * and the complicated ones may have hundreds of lines, which has certain side
 * effects on the programming experience.
 */
KernelSignature ScaleOpArgumentMapping(const ArgumentMappingContext& ctx) {
  if (ctx.IsDenseTensorInput("X")) {
    if (ctx.HasInput("ScaleTensor")) {
      return KernelSignature(
          "scale", {"X"}, {"ScaleTensor", "bias", "bias_after_scale"}, {"Out"});
    } else {
      return KernelSignature(
          "scale", {"X"}, {"scale", "bias", "bias_after_scale"}, {"Out"});
    }
  } else if (ctx.IsSelectedRowsInput("X")) {
    if (ctx.HasInput("ScaleTensor")) {
      return KernelSignature("scale_sr",
                             {"X"},
                             {"ScaleTensor", "bias", "bias_after_scale"},
                             {"Out"});
    } else {
      return KernelSignature(
          "scale_sr", {"X"}, {"scale", "bias", "bias_after_scale"}, {"Out"});
    }
  } else {
    return KernelSignature("unregistered", {}, {}, {});
  }
}
```

其中的ArgumentMappingContext基本接口设计如下：

```
// TODO(chenweihang): Add more methods if needed in future
class ArgumentMappingContext {
 public:
  virtual ~ArgumentMappingContext() = default;

  virtual bool HasInput(const std::string& name) const = 0;
  virtual bool HasOutput(const std::string& name) const = 0;
  virtual bool HasAttr(const std::string& name) const = 0;

  // now we can't use Attribute here, it will cause phi relay on
  // boost::variant and BlockDesc
  virtual paddle::any Attr(const std::string& name) const = 0;

  virtual size_t InputSize(const std::string& name) const = 0;
  virtual size_t OutputSize(const std::string& name) const = 0;

  virtual bool IsDenseTensorInput(const std::string& name) const = 0;
  virtual bool IsSelectedRowsInput(const std::string& name) const = 0;

  virtual bool IsDenseTensorOutput(const std::string& name) const = 0;
  virtual bool IsSelectedRowsOutput(const std::string& name) const = 0;
};
```

无论ScaleOpArgumentMapping是在fluid中使用，还是在infrt中使用，只要能够构造出特定框架的ArgumentMappingContext，即可获得对应的参数映射关系。

**1）对fluid的适配**

在fluid中，该函数需要同时在静态图和动态图中使用，比较直接的思路是，直接通过ExecutionContext构造ArgumentMappingContext，然后在op执行时调用，例如

```
// TODO(chenweihang): split impl based OpProto or Dygraph if needed
class ExecutionArgumentMappingContext : public phi::ArgumentMappingContext {
 public:
  ExecutionArgumentMappingContext(const ExecutionContext& ctx) : ctx_(ctx) {}

  bool HasInput(const std::string& name) const override {
    return ctx_.HasInput(name);
  }

  bool HasOutput(const std::string& name) const override {
    return ctx_.HasOutput(name);
  }

  bool HasAttr(const std::string& name) const override {
    return ctx_.HasAttr(name);
  }

  size_t InputSize(const std::string& name) const override {
    return ctx_.InputSize(name);
  }

  size_t OutputSize(const std::string& name) const override {
    return ctx_.OutputSize(name);
  }

  bool IsDenseTensorInput(const std::string& name) const override {
    return ctx_.InputVar(name)->IsType<framework::Tensor>() ||
      ctx_.InputVar(name)->IsType<framework::LoDTensor>();
  }

  bool IsSelectedRowsInput(const std::string& name) const override {
    return ctx_.InputVar(name)->IsType<framework::SelectedRows>();
  }

 private:
  const ExecutionContext& ctx_;
};
```

**2）对infrt的适配**

若在infrt中，infrt只有训练存储的推理program，也就是只有Proto这一层的信息，那么可以通过Proto信息去构造对应的Context使用，**proto中的信息目前在支持参数匹配上是完备的**，例如

```
class ProtoArgumentMappingContext : public phi::ArgumentMappingContext {
 public:
  ProtoArgumentMappingContext(proto::OpProto* op, proto::BlockDesc* block) : op_(op), block_(block) {}

  bool HasInput(const std::string& name) const override {
    // simple search
    for (int i = 0; i < proto_->input_size(); ++i) {
      auto& in = proto_->inputs()[i];
      if (in.name() == name) {
        return true;
      }
    }
    return false;
  }

  bool HasOutput(const std::string& name) const override {
    // simple search
    for (int i = 0; i < proto_->output_size(); ++i) {
      auto& out = proto_->outputs()[i];
      if (out.name() == name) {
        return true;
      }
    }
    return false;
  }

  bool HasAttr(const std::string& name) const override {
    // simple search
    for (int i = 0; i < proto_->attrs_size(); ++i) {
      auto& attr = proto_->attrs()[i];
      if (attr.name() == name) {
        return true;
      }
    }
    return false;
  }

  size_t InputSize(const std::string& name) const override {
    return proto_->input_size();
  }

  size_t OutputSize(const std::string& name) const override {
    return proto_->output_size();
  }

  bool IsDenseTensorInput(const std::string& name) const override {
    for (int i = 0; i < block_.vars_size(); ++i) {
      auto& var = block_.vars()[i];
      if (var.name() == name) {
        if (var.type() == proto::VarType::LOD_TENSOR) {
          return true;
        }
      }
    }
    // TODO(chenweihang): throw error when cannot found
    return false;
  }

  bool IsSelectedRowsInput(const std::string& name) const override {
    for (int i = 0; i < block_.vars_size(); ++i) {
      auto& var = block_.vars()[i];
      if (var.name() == name) {
        if (var.type() == proto::VarType::SELECTED_ROWS) {
          return true;
        }
      }
    }
    // TODO(chenweihang): throw error when cannot found
    return false;
  }

 private:
  proto::OpProto op_*;
  proto::BlockDesc block_*;
};
```

### 2.4.2 phi Kernel兼容调度执行

目前phi kernel可以兼容地在老Executor，ParallelExecutor，动态图的Tracer，Engine，推理的Predictor，以及新执行器InterpreterCore等在执行体系中被调度执行。

具体地，在动静态图调用OpKernel之前，判断对于当前计算，比如`scale`是否有新形式的Kernel已经注册，如果已经注册了，则调用新形式的Kernel去执行，如果没找到合适的Kernel，仍然执行之前已有的OpKernel。

```
  if (phi::KernelFactory::Instance().HasCompatiblePhiKernel(type_)) {
    if (pt_kernel_signature_ == nullptr || pt_kernel_ == nullptr) {
      pt_kernel_signature_.reset(new KernelSignature(
          std::move(GetExpectedPhiKernelArgs(exe_ctx))));
      VLOG(6) << *pt_kernel_signature_.get();

      kernel_type_.reset(
          new OpKernelType(std::move(InnerGetExpectedKernelType(exe_ctx))));
      dev_ctx = pool.Get(kernel_type_->place_);

      pt_kernel_name = pt_kernel_signature_->name;
      pt_kernel_key = TransOpKernelTypeToPhiKernelKey(*kernel_type_.get());
      pt_kernel_.reset(
          new phi::Kernel(phi::KernelFactory::Instance().SelectKernel(
              pt_kernel_name, pt_kernel_key)));

      if (pt_kernel_->IsValid()) {
        VLOG(6) << "Static mode ChoosePhiKernel - kernel name: "
                << pt_kernel_name << " | kernel key: " << pt_kernel_key
                << " | kernel: " << *pt_kernel_;
      } else {
        VLOG(6) << "Static mode ChoosePhiKernel - kernel `" << pt_kernel_name
                << "` not found.";
      }
    }
    if (pt_kernel_->IsValid()) {
      run_phi_kernel_ = true;
    } else {
      auto& all_op_kernels = AllOpKernels();
      auto kernels_iter = all_op_kernels.find(type_);
      if (kernels_iter == all_op_kernels.end() ||
          kernels_iter->second.find(*kernel_type_.get()) ==
              kernels_iter->second.end()
#ifdef PADDLE_WITH_XPU
          ||
          paddle::platform::is_xpu_place(kernel_type_->place_) &&  // NOLINT
              !paddle::platform::is_xpu_support_op(
                  type_, *kernel_type_.get())  // NOLINT
          || paddle::platform::is_in_xpu_black_list(type_)
#endif
              ) {
        auto pt_cpu_kernel_key =
            FallBackToCpu(*kernel_type_.get(), pt_kernel_key, *this);
        pt_kernel_.reset(
            new phi::Kernel(phi::KernelFactory::Instance().SelectKernel(
                pt_kernel_name, pt_cpu_kernel_key)));

        dev_ctx = pool.Get(platform::CPUPlace());
        if (pt_kernel_->IsValid()) {
          VLOG(6) << "Static mode PrepareImpl - kernel name: " << pt_kernel_name
                  << " | kernel key: " << pt_cpu_kernel_key
                  << " | kernel: " << *pt_kernel_;
          run_phi_kernel_ = true;
        }
      }
    }
  }
  if (!run_phi_kernel_) {
    if (kernel_type_.get() == nullptr || kernel_func_.get() == nullptr) {
      ChooseKernel(exe_ctx);
      dev_ctx = pool.Get(kernel_type_->place_);
    }
  }

...

    if (run_phi_kernel_) {
      phi::KernelContext pt_kernel_context;
      // Do data transform before building KernelContext
      // TODO(zhiqiu): support TransferInplaceVarsBack
      PreparePhiData(exec_scope, *pt_kernel_, *pt_kernel_signature_,
                      runtime_ctx);
      BuildPhiKernelContext(*runtime_ctx, dev_ctx, &pt_kernel_context);
      (*pt_kernel_)(&pt_kernel_context);
    } else {
      (*kernel_func_)(
          ExecutionContext(*this, exec_scope, *dev_ctx, *runtime_ctx));
    }
```

对于phi kernel的执行，有两个关键函数

**GetExpectedPhiKernelArgs**

- 在调用phi kernel时，要完成多属性到少属性的匹配，这里就需要调用前述的ArgumentMapping函数，从而得到phi kernel的参数列表，GetExpectedPhiKernelArgs实现如下：

```
KernelSignature OperatorWithKernel::GetExpectedPhiKernelArgs(
    const ExecutionContext& ctx) const {
  ExecutionArgumentMappingContext arg_mapping_ctx(ctx);
  return phi::OpUtilsMap::Instance().GetArgumentMappingFn(Type())(
      arg_mapping_ctx);
}
```

**BuildPhiKernelContext**

- 要调用phi kernel，需要准备phi kernel需要的Context，PhiKernelContext和原先的RuntimeContext及ExecutionContext不同之处在于，PhiKernelContext中是以SmallVector存储输入输出及属性，访问效率上要比原先的map高一些
- PhiKernelContext中不存储输入输出及属性的name，要求这几项顺次存储，和kernel的参数列表顺序一致

Phi KernelContext的基本设计如下：

```
/**
 * Note: KernelContext doesn't manage the life of DeviceContext and Tensor
 *
 * Note: KernelContext does not couple the concept of framework,
 *       its constructor can only take the members it needs as parameters,
 *       not Scope, RuntimeContext, etc. as parameters
 */
class KernelContext {
 public:
  KernelContext() = default;
  explicit KernelContext(DeviceContext* dev_ctx) : dev_ctx_(dev_ctx) {}

  void SetDeviceContext(DeviceContext* dev_ctx) { dev_ctx_ = dev_ctx; }

  template <typename CtxType>
  const CtxType& GetDeviceContext() const {
    return static_cast<const CtxType&>(*dev_ctx_);
  }

  void EmplaceBackInput(const TensorBase* input);

  void EmplaceBackInputWithoutSetRange(const TensorBase* input);

  void EmplaceBackInputs(paddle::small_vector<const TensorBase*> inputs);

  void EmplaceBackOutput(TensorBase* output);

  void EmplaceBackOutputWithoutSetRange(TensorBase* output);

  void EmplaceBackOutputs(paddle::small_vector<TensorBase*> outputs);

  void EmplaceBackAttr(paddle::any attr);

  const std::pair<int, int>& InputRangeAt(size_t idx) const;

  const std::pair<int, int>& OutputRangeAt(size_t idx) const;

  void AssignInputRange(std::pair<int, int>&& range, size_t idx);

  void AssignOutputRange(std::pair<int, int>&& range, size_t idx);

  template <typename TensorType>
  const TensorType& InputAt(size_t idx) const {
    return static_cast<const TensorType&>(*(inputs_.at(idx)));
  }

  template <typename TensorType>
  paddle::optional<const TensorType&> OptionalInputAt(size_t idx) const {
    const auto& input = inputs_.at(idx);
    return input ? paddle::optional<const TensorType&>{static_cast<
                       const TensorType&>(*input)}
                 : paddle::optional<const TensorType&>{paddle::none};
  }

  template <typename TensorType>
  std::vector<TensorType> MoveInputsBetween(size_t start, size_t end) {
    std::vector<TensorType> v;
    for (size_t i = start; i < end; ++i) {
      auto t = static_cast<const TensorType*>(inputs_.at(i));
      v.emplace_back(*t);
      inputs_[i] = nullptr;
    }
    return v;
  }

  template <typename TensorType>
  TensorType* MutableOutputAt(size_t idx) {
    return static_cast<TensorType*>(outputs_.at(idx));
  }

  template <typename TensorType>
  std::vector<TensorType*> MutableOutputBetween(size_t start, size_t end) {
    std::vector<TensorType*> v;
    for (size_t i = start; i < end; ++i) {
      v.emplace_back(static_cast<TensorType*>(outputs_.at(i)));
    }
    return v;
  }

  template <typename AttrType>
  AttrType AttrAt(size_t idx) const {
    try {
      return paddle::any_cast<AttrType>(attrs_.at(idx));
    } catch (paddle::bad_any_cast&) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Attribute cast error in Op Kernel Context."));
    }
  }

  size_t InputsSize() const { return inputs_.size(); }
  size_t OutputsSize() const { return outputs_.size(); }
  size_t AttrsSize() const { return attrs_.size(); }

 private:
  DeviceContext* dev_ctx_;

  paddle::small_vector<const TensorBase*> inputs_;
  paddle::small_vector<TensorBase*> outputs_;
  paddle::small_vector<paddle::any> attrs_;

  paddle::small_vector<std::pair<int, int>> input_range_;
  paddle::small_vector<std::pair<int, int>> output_range_;
};
```

## 2.5 产品思考及后续规划

目前，phi算子库仍然处在Kernel体系的建设阶段，Kernel尚未完全迁移，且仍然存在诸多完善点，但将来phi算子库会更好地将“算子”的概念纳入进来，这还需要比较长的时间和比较大的人力投入。最后，从“产品”的角度介绍一下phi后续对于算子开发范式的规划，也能够让开发者更容易理解 “为什么要做算子库重构？” 这件事。

### 2.5.1 原算子开发范式

我们应该如何描述“框架算子”这个概念？

万变不离其宗：

- 我们如何描述一个人？1. 他叫什么，长什么样；2. 他的工作、兴趣、爱好、特长、品质等
- 我们如何描述一个物品？1. 它叫什么，长什么样；2. 它的用途和功能是什么
	- 比如一个杯子：1. 它叫水杯，长这样；2. 它用来盛水的

简单说，我们描述一个对象，可以采用两段式结构：

1. 它的名字，样子或者说形态
2. 它的功能，特征，及细节

算子同样可以按照这个原则去类比：

1. 这个算子叫什么，有哪些参数，返回值是什么（即Op）
2. 这个算子在不同场景，不同设备中，怎么执行，怎么计算（即Kernel）

如果我们**能分得清楚1和2的边界，并守住这个边界**，我们设计就能够趋于简练。

这是什么意思？就是说如果我们一定要用两段式来介绍一个对象，那么哪部分应该在第一段？哪部分应该在第二段？得有个逻辑清晰的认知。例如，我们用两段式介绍一个人：

- 方式1：1. 他叫张三；2. 他在百度工作，他喜欢唱歌、爬山、骑行，他待人真诚，认真负责
- 方式2：1. 他叫张三，他喜欢唱歌；2. 他在百度工作，他喜欢爬山、骑行，他待人真诚，认真负责

哪种分段方式更好一些呢？答案是显然的，方式2的两段中有同样形式的内容，逻辑不清。

为什么用这种方式来类比？因为我们的算子开发面临的场景我觉得是一样的，市面上现有的框架，对于算子的定义，都围绕着**“1. 算子描述，2. 算子执行”**的两段式进行设计。

顺着这个思路，我从“语文、逻辑和信息认知”的角度介绍一下我对fluid算子开发现状的理解，如果把现在的算子体系当做一篇以**算子**为题目的“小学作文”来看的话，拿高分有点困难。

**（1）"生僻词"比较多**

fluid的Op开发概念对于新人来讲，可能是一种看“文言文”的感觉，似懂非懂。

如果我要描述一个“运算”，我需要讲清楚它叫什么，输入输出有哪些，这就够了，例如一个乘法运算，`叫multiply，输入x，y，得到out`，在这一点上，Python API是足够简练的。

那么现在我们的内部算子要怎么描述呢？要实现以下类和函数，可以参考 [mul_op](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/operators/mul_op.cc) ：

```
# 前向op
OperatorWithKernel
- InferShape
- GetExpectedKernelType
- GetKernelTypeForVar
OpMaker
# 反向op
GradOp
GradOpMaker
# 类型推导
VarTypeInference
# 显存优化
InplaceOpInference
NoNeedBufferVarsInference
# Op注册
REGISTER_OPERATOR
# Op版本管理
REGISTER_OP_VERSION
```

直观看说实话会有点困惑，新人可能会有什么疑问呢？

- Operator可以理解，是算子，OpMaker也可以理解，是告诉这个算子怎么生成，但为什么要两个呢？OpMaker都已经告诉你要怎么生成这个Op了，为什么还需要再写一个Operator？Maker不应该把Operator make出来吗？
- 除了这俩，剩下的都看不懂。。。这些都什么意思？什么时候用？我新开发这个算子哪个需要写，哪个不需要写？
- 算了，我短时间内搞不懂，算子也着急，找一个类似的算子，copy一份，它写了什么我照着挪过来，能跑对就行。。。（这可能是大部分新人开发算子的心态）

**（2）重复的“修饰”比较多**

其实每个算子真正不同的信息，就那么几个词，剩余的东西都是模板一样的存在，人脑处理信息是有成本的，去区分差异是需要思考的，从产品的角度来讲，直接把不同的地方告诉用户，让用户只关注这些差异是最高效的。

以 [MulOpMaker](https://github.com/PaddlePaddle/Paddle/blob/c6f49f0b9f189e043b458348d7fd1468e2645621/paddle/fluid/operators/mul_op.cc#L72) 和 [DotOpMaker](https://github.com/PaddlePaddle/Paddle/blob/c6f49f0b9f189e043b458348d7fd1468e2645621/paddle/fluid/operators/dot_op.cc#L35) 的实现为例，我们可以发现以下几点：

1. 除了Op名字，输入、输出和参数命名，两段结构极其类似？为什么我们不能把这几个空抠出来让开发者直接填空？
2. 输入、输出和参数后面的大段描述属于**重复建设**，并且现阶段没有用处，因为Python端已经写过一遍了，并且写得更规范，更清楚，C++端这里的参数注释没有人把关，质量参差不齐。

再看Operator的GetExpectedKernelType（[mul](https://github.com/PaddlePaddle/Paddle/blob/c6f49f0b9f189e043b458348d7fd1468e2645621/paddle/fluid/operators/mul_op.cc#L41)和[dot](https://github.com/PaddlePaddle/Paddle/blob/c6f49f0b9f189e043b458348d7fd1468e2645621/paddle/fluid/operators/dot_op.cc#L28)）：，也一样，都是根据x选择kernel，那为什么还要让开发者写其他的内容呢？直接做个填空，填x是不是就行了。

我们开发Op的时候，这些组件多少都存在这样的问题，这增加了大家工作量和理解成本。

**（3）相同的“段落”写了好多遍**

这里主要指OpKernel的开发，我们现在的OpKernel之间可复用性较差，比如已经有了mul和add的Kernel，我们现在要新增一个fc算子（由mul和add）组成，我们得去mul和add的kernel中把代码拷贝一份过来使用，而不能直接调用mul和add的kernel。

这是我们建设phi初期要解决的问题，并且我从周围新人的口中已经听到过多次这样的反馈：

- 我开发新算子，需要一个broadcast操作，我得去另一个算子里copy过来，还得先调通，copy的时候可能没copy全，或者应用场景稍有不同，这都需要额外的时间
- 实现gumbol-softmax算子，因为softmax是其中的子运算，我得先把softmax的kernel实现copy过来

**（4）“描述”本身有二义性分段**

说回开始的两段式结构，”1.算子描述；2.算子执行“，分这两段是必要的，也是业界普遍的做法，我们不需要再分第三段了，但paddle目前存在第三段，算子描述分了两段进行，并且这两段还不一致，即PythonaAPI和Op。

API和Op都是对算子运行行为的概要描述，本质上只是同一段内容的不同展现形式，比如[Python dot API](https://github.com/PaddlePaddle/Paddle/blob/c6f49f0b9f189e043b458348d7fd1468e2645621/Python/paddle/tensor/linalg.py#L993)和[DotOpMaker](https://github.com/PaddlePaddle/Paddle/blob/c6f49f0b9f189e043b458348d7fd1468e2645621/paddle/fluid/operators/dot_op.cc#L35)，就是告诉别人“它叫什么，参数都是什么”。


咱们对同一个东西的描述，分两个地方写，还写得不一样，这是很令人费解的。就好像你介绍一个人，在学校你说他叫”张三“，在公司你说他叫”张三丰“，有相像之处，但又不是一个意思。

对于一个算子，它的输入、输出应该在各个场景下都是一致的，如果不一致，那本质上就不是一个算子。

比如，conv2d的api和op，[Python conv2d API](https://github.com/PaddlePaddle/Paddle/blob/c6f49f0b9f189e043b458348d7fd1468e2645621/Python/paddle/nn/functional/conv.py#L416)，很简单，8个输入参数；但是对应的[conv2d op](https://github.com/PaddlePaddle/Paddle/blob/c6f49f0b9f189e043b458348d7fd1468e2645621/paddle/fluid/operators/conv_op.cc#L259)，有**32个**输入参数，让人摸不着头脑。

开发者也会很困惑，我开发op的时候，API和Op不是一个东西吗，我应该写得一样呢？还是不一样？

推理之前为什么要做**算子增强推全**，就是op的参数太多了，但API的参数很少，这两者本来是介绍一个东西，却差别如此之大，所以需要发动全员，在op的某些参数上标记AsExtra，就声明这个参数可能是多余的。

当然我们演变到如此田地，有一定历史原因：

1. Op输入输出参数规范限制差，留的口子太大，可以天马行空地写；
2. 2.0 API对外层Python API的形态做了大范围规整，但是Op层保持不变，是导致目前同一段描述差异变大的一个主要原因。

对于这个问题的解决，我们的方向是很明确的，就是**Op层描述向API层靠拢，因为API层的定义是经过2.0 API项目仔细设计过的**。

### 2.5.2 新算子开发范式：完形填空 + 拼积木

phi期望的Op开发方式：**“完形填空”式算子描述实现 + “堆积木”式算子执行实现**

**Op实现：**

需要写的内容如下：

```
# 配置文件 api.yaml
- api : add
  args : (const Tensor& x, const Tensor& y)
  output : Tensor
  infer_meta :
    func : ElementwiseInferMeta
    param : [x, y, -1]
  kernel :
    func : add
    param : [x, y, -1]
```

以填空为主要方式，名字，输入、输出、输出的增强推断，用什么Kernel。

原先需要写得大段重复代码，全部通过”代码自动生成“的手段去实现，开发者不用再关注。

主要思想：仅让开发者关注最小的差异化信息集合，填空指定信息。

这里Op配置时，要求和Python端参数命名等完全一致，做到上下层描述一致，不给开发者留空间在op层自由发挥，导致想加什么加什么的随意行为。如果需要给op加参数，API也要一起更新，这首先需要通过不兼容升级评审。

**Kernel实现：**

```
template <typename T, typename Context>
Fc(const Context& dev_ctx, const Tensor& x, const Tensor& w, const Tensor& b, Tensor* out) {
	phi::add<T, Context>(phi::mul<T，Context>(x, w), b, out);
}

PT_REGISTE_KERNEL("fc", Fc, ...)
```

mul和add操作的拼接，代码量很少，再加一个注册声明。

整个Op+Kernel的开发也就十几行代码，在去除所有冗余信息，仅保留差异化信息上，这种方式已经是没有什么精简空间了。
