## 一、FP16简介及精度问题

### 1.1 FP16简介

FP16指得是半精度浮点数表示,通常意义上其表示的为Nvidia提供的半精度浮点数表示方案， 也被IEEE 754-2008方案所采纳。此方案有别于Intel提供的半精度表示方案BF16, BF16的采用直接截断尾数的方式。FP16较BF16拥有更长的尾数，但阶码较短。因此FP16可以提供较BF16更好的有效位长度，而BF16可以提供较FP16更广的动态范围。

**表1-1 FP16和BF16在近1端的精度表现**

| Format | Epsilon(ε) |
| ------ | ----------- |
| FP32   | 0.00000012  |
| FP16   | 0.00097656  |
| BF16   | 0.00781250  |

注：Epsilon是各浮点表示形式下使得1+ε >1成立的最小浮点数值

**表1-2 FP16、FP32与BF16的动态范围**

| Format | Range             |
| ------ | ----------------- |
| FP32   | 1.4E-45～3.40E38  |
| FP16   | 5.96E−8 ~ 655    |
| BF16   | 9.2E−41～3.39E38 |

### 1.2 FP16格式

FP16格式采用16位对浮点数进行表示，其中尾数位为10位，阶码为5位，符号位1位。

BF16格式采用16位对浮点数进行表示，其中尾数位为7位，阶码为8位，符号位1位。

FP16格式与BF16、FP32格式的比较见表1-3。

注意阶码部分采用移码表示，偏移值为15。即01111(2) = 0(10)。如此方便进行FP16数值大小比较。

**表1-3 FP16与BF16、FP32格式的对比**

| Format | Bits | Exponent | Fraction |
| ------ | ---- | -------- | -------- |
| FP32   | 32   | 8        | 23       |
| FP16   | 16   | 5        | 10       |
| BF16   | 16   | 8        | 7        |

注： 表中数字均表示相应项的位数长度

![](./images/data_format.png)

![](./images/data_range.png)

### 1.3 FP16/BF16的数值范围及精度

浮点数的表示方案不在本节表述之内，请自行查阅，可参考[IEEE 754相关介绍](https://en.wikipedia.org/wiki/IEEE_754)，在此不做赘述。FP16可表示数据的最大正值为65504，超过此数值会造成上溢出问题，舍入为+INF；最小正值约为0.0000000596（非规格表示下），低于此数值会造成下溢出问题，舍入为+0。符号为负时以绝对值为参考，分别在满足各自情况下舍入为-INF与-0。关于各个值的表示可见表1-4 。

**表1-4 FP16的各特殊值表现**

| 二进制格式（符号 阶码 尾数） | 数值（十进制）    | 备注     |
| ---------------------------- | ----------------- | -------- |
| X 00000 0000000000           | ±0               |          |
| X 11111  0000000000          | ±INF             |          |
| X 11111  XXXXXXXXXX(非全0)   | Nan               |          |
| 0 00000  0000000001          | 0.000000059604645 | 最小正值 |
| 0 11110 0000000000           | 65504             | 最大正值 |

注：X 指0或1

**表1-5 BF16的各特殊值表现**

| 二进制格式                  | 数值（十进制）        | 备注     |
| --------------------------- | --------------------- | -------- |
| X 00000000 0000000          | ±0                   |          |
| X 11111111 0000000          | ±INF                 |          |
| X 11111111 XXXXXXX（非全0） | NAN                   |          |
| 0 11111110 1111111          | $3.38953139×10^{38}$ | 最大正值 |
| 0 00000000 0000001          | 9.2 × 10−41         | 最小正值 |

注：X 指0或1

通常意义上，FP16可表示十进制下的三位有效数字格式($\log_{10}^{2^{11}}\sim3.311$) ，BF16可表示十进制下的2位有效数字格式($\log_{10}^{2^{8}}\sim2.408$)，但按照浮点数的格式，其表示的绝对精度（精确到哪一数位）在各个区间（本质为各个阶码下并不相同)。

### 1.4 FP16/BF16在计算中的精度问题

#### 1.4.1 溢出问题

1. FP16的可表示范围较FP32等更小，容易触发上、下溢出问题。

关于FP16的上下溢出可以参考图1-3。对于绝对值大于65504的数，触发上溢出会舍入到±INF；对于绝对值小于的数，触发下溢出舍入到0，具体示例可参考表1-6。

**表1-6 FP16格式下数据表示示例**

![](./images/overflow.png)

1. BF16则因为阶码同FP32等长，因此并不容易出现上下溢出问题。

#### 1.4.2 舍入问题。

1. FP16格式的浮点数最多只能表示3位有效数字，所以各浮点区间的固定间隔都是 $Interval=Min*2^{-10}$。

   1. 因此当 $(\frac{累加值}{加数})>2^{11}$ 时，计算结果超过了3位有效数字；这会造成累加值最终无法被有效表示，累加结果会被舍入到累加值本身。

      1. 数值: $1+0.0001=1.0001$
      2. FP16: $1+0.0001=1.0001(无法表示->舍入->1)$
   2. FP32格式拥有7位有效数字的表达效果，因此当FP32格式向FP16格式转化时，也会出现精度的舍入问题。

      1. FP32：0.1234567
      2. FP16：0.1235
2. BF16格式的浮点数，在各个区间内的固定间隔是 $Interval=Min*2^{-7}$ ，故BF16相较于FP16的精度更低，也更容易出现FP16中所阐述的计算舍入和转换精度丢失的问题。

## 二、FP16/BF16算子开发规范及示例代码

对于添加算子的流程具体可参考[算子添加流程](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_cpp_op_cn.html#python-api)，当我们向Kernel中添加FP16或BF16数据类型支持时，需要关注流程中如下4个部分。

### **2.1 查找对应的实现文件**

1. 检索对应Kernel的**头文件**

   1. 通常可以在paddle/phi/kernels/ 目录下检索到与具体硬件无关的Kernel源代码文件及头文件。
   2. 检索方式。

      1. paddle/phi/kernels/ 目录下的文件名通常形式为 Kernel名字 + '\_kernel' 。因此寻找某个具体Kernel的相关文件时，可以通过文件名快速查找到。如Abs的相关文件为abs_kernel.h
      2. 可以利用grep，或者IDE中带有的查找功能等对Kernel名本身进行搜索，定位具体的实现文件。
2. 检索对应Kernel的 **源文件** 。

   1. 按照需要提供支持的硬件，在相应的目录下查找对应Kernel的源文件。Kernel的GPU版本实现代码存放在paddle/phi/kernels/gpu/ 目录下，CPU版本实现代码存放在paddle/phi/kernels/cpu/ 目录下。

      1. FP16数据类型一般在GPU、XPU上支持，可在gpu、xpu目录下的相关文件中添加FP16数据支持。
      2. BF16数据类型一般在CPU、GPU上均有支持，可在cpu、gpu目录下的相关文件中添加FP16数据支持。
   2. 检索方式。

      1. 同1.b.i相同，采用文件名进行查找。如Abs相关文件为abs_kernel.cu

### **2.2 引入所需的头文件**

为了使用FP16/BF16数据类型，需要在GPU源文件中引入下列头文件之一。其中3中定义了FP16/BF16数据类型，1包含2包含3。引入这些文件后，可以使用phi::dtype::float16来表示FP16数据类型，可以使用phi::dtype::bfloat16来表示BF16数据类型。

1. paddle/phi/core/device_context.h
2. paddle/phi/common/data_type.h
3. 1. FP16数据类型：paddle/phi/common/float16.h
4. BF16数据类型：paddle/phi/common/bfloat16.h

### **2.3 在Kernel注册处添加FP16/BF16数据类型支持**

每一个GPU源文件的末尾，都有用来进行Kernels注册的宏 PD_REGISTER_KERNEL。如Abs的注册宏代码2-1。

```cpp
PD_REGISTER_KERNEL(abs,
                   GPU,
                   ALL_LAYOUT,
                   phi::AbsKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
```

其中phi::AbsKernel是abs对应的具体GPU实现函数，在这之后的float, double, int, ... 等为abs所支持的数据类型。因此要扩充FP16数据类型支持，需要在该处添加phi::dtype::float16数据类型。如果需要扩充BF16数据类型的支持，可添加phi::dtype::bfloat16。

### **2.4 函数中添加FP16/BF16数据实现的特化支持**

在注册时添加FP16/BF16的数据类型支持后，在对应的实现中也应添加对应的支持。

在Paddle中，各个Kernel的实现采用模版函数实现，函数所支持的数据类型也以模版参数的形式进行传入。因此可以自行编译出支持FP16/BF16实现的代码。但为了提升FP16/BF16计算的精度，在涉及数学计算函数和归约计算时，需要我们对FP16/BF16算子进行特化。

#### 2.4.1 数学计算函数

数学计算函数是对于输入按照数学运算规则进行求解。Paddle中的实现的大部分基本数学算子最终均归结于对Math库中的数学函数的调用。关于数学计算函数在FP32和FP16下的计算结果误差对比可参考附录。

##### 2.4.1.1 存在问题

1. 部分函数在FP16/BF16下相比FP32误差较高。典型的数学函数有reciprocal、exp、expm1、tan、cosh、sinh、square、tanh_shrink。
2. 部分函数会产生较高的数值输出，FP16容易造成上溢出。典型的数学函数有exp、expm1、square、tan、atanh、cosh、sinh。

##### 2.4.1.2 特化实现代码

数学计算函数通常实现时， **需要将FP16/BF16格式转换为FP32格式，然后利用FP32作为输入计算数值结果，后再将结果转换为FP16/BF16格式作为输出** 。以Cos函数的实现为例。

```cpp
template <typename T, typename Context, typename Functor>
void ActivationGPUImpl(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out,
                       const Functor& functor) {
  PADDLE_ENFORCE_NOT_NULL(out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
}

template <typename T>
struct CudaCosFunctor {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  __device__ __forceinline__ T operator()(const T arg_x) const {
    MPType x = static_cast<MPType>(arg_x);
    return static_cast<T>(cos(x));
  }
};

template <typename T, typename Context>
void CosKernel(const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) {
	funcs::CudaCosFunctor<T> functor;
	ActivationGPUImpl<T, Context, funcs::CudaCosFunctor<T>>(
        dev_ctx, x, out, functor);
}
```

* 关于MPTypeTrait的定义可参考附录

#### 2.4.2 归约计算

归约计算往往会设计大量元素参与运算，在这种情况下，很容易出现误差的累积，对于低精度计算并不友好。

##### 2.4.2.1 存在问题

具有较多输入参数的大规模归约场景对浮点数的精度影响较大，主要有两个方面。

1. 受限于FP16/BF16的低精度，大规模浮点归约的后期，容易形成“截断上界”，导致频繁的浮点数舍入行为。
2. 大规模浮点规约因为浮点计算次数较多，容易累积误差，使得误差越来越大。

##### 2.4.2.2 特化函数实现

最好 **使用FP32格式进行归约。** 以add_n为例，增添转换为FP32数据类型进行归约的一个建议的实现框架可以参考代码2-3。

```cpp
template <class T>
__global__ void SumArrayCUDAKernel(
    T **in, T *out, int64_t N, size_t in_size, bool read_dst) {
  using MPType = typename phi::dtype::MPTypeTrait<T>::Type;

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    MPType total(read_dst ? static_cast<MPType>(out[id]) : static_cast<MPType>(0));
    for (int i = 0; i < in_size; ++i) {
      const T *tmp = in[i];
      if (tmp) {
        total += static_cast<MPType>(tmp[id]);
      }
    }
    out[id] = static_cast<T>(total);
    id += blockDim.x * gridDim.x;
  }
}

template <typename T, typename Context>
void AddNKernel(const Context &dev_ctx,
                const std::vector<const TensorBase *> &x,
                DenseTensor *out) {
  ...... //pre-dealing

	SumArrayCUDAKernel<T><<<grids, blocks, 0, stream>>>(in_array_data,
                                                        out->data<T>(),
                                                        lod_length,
                                                        in_data.size(),
                                                        dst_write | in_place);

	...... //post-dealing
}
```

* 关于MPTypeTrait的定义可参考附录

## 三、附录

### 3.1 关于MPTypeTrait实现

MPtypeTrait可以按照代码3-1进行实现，在传入FP16/BF16类型时，在类内部定义FP32类型。

源代码链接：[https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/amp_type_traits.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/common/amp_type_traits.h)

```cpp
template <typename T>
class MPTypeTrait {
  public:
    using Type = T;
};

template <>
class MPTypeTrait<phi::dtype::float16> {
public:
  using Type = float;
};

template <>
class MPTypeTrait<phi::dtype::bfloat16> {
public:
  using Type = float;
};
```

### 3.2 关于__hisinf和__hisnan

#### 3.2.1 __hisinf

```cpp
__device__ int __hisinf(const __half a)
```

定义：检查输入的half类型的参数是否为INF

参数：half类型，只读属性

返回：int类型，其中0表示非INF，1表示+INF，-1表示-INF

#### 3.2.2 __hisnan

```cpp
__device__ bool __hisnan ( const __half a )
```

定义：检查输入的half类型的参数是否为NAN

参数：half类型，只读属性

返回：bool类型，true表示是NAN，false表示非NAN

### 3.3 基本数学函数在FP32和FP16的误差对比

**基本数学计算函数在FP16格式下与FP32格式下计算结果的误差**

| 算子                                                  | 最大绝对误差       | 最大相对误差      | 平均绝对误差      | 平均相对误差      |
| ----------------------------------------------------- | ------------------ | ----------------- | ----------------- | ----------------- |
| sin/cos/atan/acosh/asinh/tanh/acos/asin/atanh         | 0.000244~0.003905  | 0.000486~0.000517 | 5.85E-05~0.001138 | 0.000115~0.000179 |
| log2/log10/log1p                                      | 0.001953~0.0033905 | 0.000482~0.000487 | 0.0004~0.001258   | 0.000171~0.000175 |
| floor/round/relu/ceil                                 | 0                  | 0                 | 0                 | 0                 |
| sqrt/rsqrt                                            | 0.031227~0.062485  | 0.000488          | 0.002071~0.004158 | 0.000172~0.000179 |
| exp/expm1/tan/cosh/sinh/reciprocal/square/tanh_shrink | 1~16               | 0.000486~1        | 0.000786~0.378427 | 9.18E-05~0.082869 |
| logsigmoid/sigmoid/silu/softsign                      | 0.000701~0.002665  | 0.000486~0.000945 | 6.55E-05~9.96E-05 | 0.000104~0.000203 |

注：

* 其中我们将数学函数归类为三角函数，对数函数，舍入函数，指数函数进行了测试，其中误差相对较大的exp/expm1（红色标注）函数等我们将其提取出来放在了一组，另外其他一些未归类的sigmoid等误差相对较低也同一成了一组
* 以上测试均基于Paddle的Python模块进行

### 3.4 基本数学函数在FP32和FP16的误差对比

1. FP32转FP16的统计误差：

   1. 包含非规规格情况下($2^{-24}\sim65504$)：

| 最大绝对误差 | 最大相对误差 | 平均绝对误差 | 平均相对误差 |
| ------------ | ------------ | ------------ | ------------ |
| 16           | 0.33289      | 0.38871      | 0.008834     |

1. 1. 仅规格化数情况下($2^{-14}\sim65504$)：

| 最大绝对误差 | 最大相对误差 | 平均绝对误差 | 平均相对误差 |
| ------------ | ------------ | ------------ | ------------ |
| 16           | 0.00048      | 0.51876      | 0.000175     |

注：

* 绝对误差采用FP32和转换后的FP16的距离
* 相对误差采用FP32和转换后的FP16的距离相对于FP32的比重

1. FP32转FP16理论上的最大相对误差可参考实数转FP16的情况
2. 实数转FP16在理论上单次的最大相对舍入误差（规格化下）: $2^{-11}$
3. 实数转FP32在理论上单次的最大相对舍入误差（规格化下）: $2^{-24}$
4. 实数转FP16在理论上单次的最大相对舍入误差（非规格化下）：0.5
5. 实数转FP32在理论上单次的最大相对舍入误差（非规格化下）：0.5

延伸：FP16和FP32精度对比时，应当采用相对误差为度量尺度，小于最大相对误差，相对误差可以保证在一定的范围内，但是绝对误差的范围本身的差距太大了，并不适合做度量尺度。

## 四、参考文章

1、[半精度浮点格式](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)

2、[IEEE 754相关介绍](https://en.wikipedia.org/wiki/IEEE_754)

3、[BF16、FP16、FP32比较](https://www.johndcook.com/blog/2018/11/15/bfloat16/)

4、[Nivdia混合精度训练手册](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#framework)

5、[随机舍入介绍](https://nhigham.com/2020/07/07/what-is-stochastic-rounding/)

6、[Nvidia 半精度接口手册](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH____HALF__COMPARISON.html)
