##ElementwiseUnary

###函数定义

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
__device__ void ElementwiseUnary(OutT* out, const InT* in, OpFunc compute)；
```

###函数说明
按照 compute 中的计算规则对 in 进行计算，将计算结果按照 OutT 类型存储到寄存器 out 中

###模板参数
```
InT : 输入数据的类型。
OutT : 存储到out寄存器中的类型。
NX : 每个线程需要计算 NX 列数据。
NY : 每个线程需要计算 NY 行数据。
BlockSize : 设备属性，标识当前设备线程索引方式。对于GPU，threadIdx.x 用作线程索引，对于 XPU，core_id() 用作线程索引。
OpFunc : 计算函数，定义方式如下：
  template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT& a) const {
      return ...;
    }
  };

```

###函数参数

```
out : 输出寄存器指针，大小为 NX x NY。
in : 输入寄存器指针，大小为 NX x NY。
compute : 计算函数，声明为OpFunc<InT，OutT>()。
```

##ElementwiseBinary
###函数定义

```

template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
__device__ void ElementwiseBinary(OutT* out, const InT* in1, const InT* in2, OpFunc compute)；
```

###函数说明
按照 compute 中的计算规则对i n1、in2 进行计算，将计算结果按照 OutT 类型存储到寄存器 out 中。

###模板参数
```
InT : 输入数据的类型。
OutT : 存储到out寄存器中的类型。
NX : 每个线程需要计算 NX 列数据。
NY : 每个线程需要计算 NY 行数据。
BlockSize : 设备属性，标识当前设备线程索引方式。对于GPU，threadIdx.x 用作线程索引，对于 XPU，core_id() 用作线程索引。
OpFunc : 计算函数，定义方式如下：
  template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT& a, const InT& b) const {
      return ...;
    }
  };

```

###函数参数

```
out : 输出寄存器指针，大小为 NX x NY。
in1 : 左操作数寄存器指针，大小为 NX x NY。
in2 : 右操作数寄存器指针，大小为 NX x NY。
compute : 声明为OpFunc<InT，OutT>()的计算对象。
```

##CycleBinary

###函数定义

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
__device__ void CycleBinary(OutT* out, const InT* in1, const InT* in2, OpFunc compute)；

```

###函数说明
按照 compute 中的计算规则对 in1、in2 进行计算，将计算结果按照 OutT 类型存储到寄存器 out 中. in1 的 shape 为[1, NX], in2 的 shape 为 [NY, NX]，实现in1， in2的循环计算，out的shape是[NY, NX]。

###模板参数
```
InT : 输入数据的类型。
OutT : 存储到out寄存器中的类型。
NX : 每个线程需要计算 NX 列数据。
NY : 每个线程需要计算 NY 行数据。
BlockSize : 设备属性，标识当前设备线程索引方式。对于GPU，threadIdx.x 用作线程索引，对于 XPU，core_id() 用作线程索引。
OpFunc : 计算函数，定义方式如下：
  template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT& a, const InT& b) const {
      return ...;
    }
  };

```

###函数参数

```
out : 输出寄存器指针，大小为 NX x NY。
in1 : 左操作数寄存器指针，大小为 NX。
in2 : 右操作数寄存器指针，大小为 NX x NY。
compute : 声明为OpFunc<InT，OutT>()的计算对象。
```



##ElementwiseTernary

###函数定义

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
 __device__ void ElementwiseTernary(OutT* out, const InT* in1, const InT* in2, const InT* in3, OpFunc compute)；

```

###函数说明
按照 compute 中的计算规则对 in1、in2、in3 进行计算，将计算结果按照OutT类型存储到寄存器out中。

###模板参数
```
InT : 输入数据的类型。
OutT : 存储到out寄存器中的类型。
NX : 每个线程需要计算 NX 列数据。
NY : 每个线程需要计算 NY 行数据。
BlockSize : 设备属性，标识当前设备线程索引方式。对于GPU，threadIdx.x 用作线程索引，对于 XPU，core_id() 用作线程索引。
OpFunc : 计算函数，定义方式如下：
  template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT& a, const InT& b, const InT& c) const {
      return ...;
    }
  };
```

###函数参数

```
out : 输出寄存器指针，大小为 NX x NY。
in1 : 操作数1的寄存器指针，大小为 NX x NY。
in2 : 操作数2的寄存器指针，大小为 NX x NY。
in3 : 操作数3的寄存器指针，大小为 NX x NY。
compute : 声明为OpFunc<InT，OutT>()的计算对象。
```

##ElementwiseAny

###函数定义

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, int Arity, class OpFunc>
__device__ void ElementwiseAny(OutT* out, InT (*ins)[NX * NY],
                                               OpFunc compute);
```

###函数说明
按照 compute 中的计算规则对 ins 中的输入进行计算，将计算结果按照OutT类型存储到寄存器 out 中，所有输入输出的维度相同。

###模板参数
```
InT : 输入数据的类型。
OutT : 存储到out寄存器中的类型。
NX : 每个线程需要计算 NX 列数据。
NY : 每个线程需要计算 NY 行数据。
BlockSize : 设备属性，标识当前设备线程索引方式。对于GPU，threadIdx.x 用作线程索引，对于 XPU，core_id() 用作线程索引。
Arity: 指针数组 ins 中指针个数。
OpFunc : 计算函数，定义方式如下：
template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT* args) const {
      return ...;
    }
  };

```
###函数参数

```
out : 输出寄存器指针，大小为 NX x NY。
ins : 由多输入指针构成的指针数组，大小为Arity.
compute : 声明为OpFunc<InT，OutT>()的计算对象。
```

##Reduce

###函数定义

```
template <typename T, int NX, int NY, int BlockSize, class ReduceFunctor, details::ReduceMode Mode>
__device__ void Reduce(T* out, const T* in, ReduceFunctor reducer, bool reduce_last_dim);
```

###函数说明
根据 reducer 对 in 中的数据进行数据规约，in数据size为[NY， NX]，当 ReduceMode = kLocalMode 时，对 in 沿着 NX 方向进行规约，完成线程内规约，out为[NY, 1]；当 ReduceMode = kGlobalMode 时，使用共享内存完成 block 内线程间的规约操作，in 和 out 的size相同，均为[NY, NX]。

###模板参数
```
T : 输入数据的类型。
NX : 每个线程需要计算 NX 列数据。
NY : 每个线程需要计算 NY 行数据。
BlockSize : 设备属性，标识当前设备线程索引方式。对于GPU，threadIdx.x 用作线程索引，对于 XPU，core_id() 用作线程索引。
ReduceFunctor : Reduce计算函数，定义方式如下：
  template <typename InT>
  struct XxxFunctor {
     HOSTDEVICE OutT operator()(const InT& a, const InT& b) const {
       return ...;
     }
  };

Mode: 规约模式，可以取值为 kGlobalMode、kLocalMode，当 ReduceMode = kLocalMode 时，对 in 沿着 NX方 向进行规约操作，完成线程内 reduce。当 ReduceMode = kGlobalMode 时,使用共享内存完成 block 内线程间的规约操作。

```

###函数参数
```
out : 输出寄存器指针，大小为 NX x NY。
in : 输入寄存器指针，大小为 NX x NY。
reducer: 规约方式，可以使用ReduceFunctor<InT>()进行定义。
reduce_last_dim: 表示原始输入的最后一维是否进行规约。
```
