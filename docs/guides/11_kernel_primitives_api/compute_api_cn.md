##ElementwiseUnary

###函数定义

```
 template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
 __device__ __forceinline__ void ElementwiseUnary(OutT* out, const InT* in, OpFunc compute)；
```
###函数说明
按照compute中的计算规则对in进行计算，将计算结果按照OutT类型存储到out中
###模板参数
```
InT: 输入数据的类型
OutT：输出数据的类型
NX：有NX列数据参与计算
NY：有NY行数据参与计算
BlockSize：设备属性，标识当前设备线程索引方法。对于GPU，threadIdx.x用作线程索引，而对于xpu，core_id（）用作线程索引。
OpFunc: 计算函数，定义方式如下：
  template <typename InT, typename OutT>
  struct XxxFunctor {
  HOSTDEVICE OutT operator()(const InT& a) const {
    return ...;
  }
};

```
###函数参数

```
out：out的寄存器指针，大小为NX*NY。
in：in的寄存器指针，大小为NX*NY。
compute：声明为OpFunc<InT，OutT>（）的计算。
```

##ElementwiseBinary
###函数定义

```

template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
__device__ __forceinline__ void ElementwiseBinary(OutT* out, const InT* in1,
                                                          const InT* in2,
                                                                                                            OpFunc compute)；
```
###函数说明
按照compute中的计算规则对in1、in2进行计算，将计算结果按照OutT类型存储到out中。

###模板参数
```

InT: 输入数据的类型
OutT：输出数据的类型
NX：有NX列数据参与计算
NY：有NY行数据参与计算
BlockSize：设备属性，标识当前设备线程索引方法。对于GPU，threadIdx.x用作线程索引，而对于xpu，core_id（）用作线程索引。
OpFunc: 计算函数，定义方式如下：
  template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT& a, const InT& b) const {
      return ...;
    }
  };

```
###函数参数

```
out：out的寄存器指针，大小为NX*NY。
in1：in1的寄存器指针，大小为NX*NY。
in2：in2的寄存器指针，大小为NX*NY。
compute：声明为OpFunc<InT，OutT>()的计算对象。
```

##ElementwiseTernary

###函数定义

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
 __device__ __forceinline__ void ElementwiseTernary(OutT* out, const InT* in1, const InT* in2, const InT* in3, OpFunc compute)；

```
###函数说明
按照compute中的计算规则对in1、in2进行计算，将计算结果按照OutT类型存储到out中
###模板参数
```
InT: 输入数据的类型
OutT：输出数据的类型
NX：有NX列数据参与计算
NY：有NY行数据参与计算
BlockSize：设备属性，标识当前设备线程索引方法。对于GPU，threadIdx.x用作线程索引，而对于xpu，core_id（）用作线程索引。
OpFunc: 计算函数，定义方式如下：
  template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT& a, const InT& b, const InT& c) const {
      return ...;
    }
  };
```
###函数参数

```
out：out的寄存器指针，大小为NX*NY。
in1：in1的寄存器指针，大小为NX*NY。
in2：in2的寄存器指针，大小为NX*NY。
in3：in3的寄存器指针，大小为NX*NY。
compute：声明为OpFunc<InT，OutT>（）的计算。
```

##ElementwiseAny

###函数定义

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, int Arity, class OpFunc>
__device__ __forceinline__ void ElementwiseAny(OutT* out, InT (*ins)[NX * NY],
                                               OpFunc compute);
```
###函数说明
按照compute中的计算规则对in进行操作，将计算结果按照OutT类型存储到out中，所有输入输出的维度相同。

###模板参数
```
InT: 输入数据的类型
OutT：输出数据的类型
NX：有NX列数据参与计算
NY：有NY行数据参与计算
BlockSize：设备属性，标识当前设备线程索引方法。对于GPU，threadIdx.x用作线程索引，而对于xpu，core_id（）用作线程索引。
Arity: ins中输入指针个数
OpFunc: 计算函数，定义方式如下：
template <typename InT, typename OutT>
  struct XxxFunctor {
    HOSTDEVICE OutT operator()(const InT* args) const {
      return ...;
    }
  };

```
###函数参数

```
out：out的寄存器指针，大小为NX*NY。
ins：由多输入指针构成的指针数组
compute：声明为OpFunc<InT，OutT>（）的计算。
```

##Reduce

###函数定义

```
template <typename T, int NX, int NY, int BlockSize, class ReduceFunctor, details::ReduceMode Mode>
__device__ __forceinline__ void Reduce(T* out, const T* in, ReduceFunctor reducer, bool reduce_last_dim);
```
###函数说明
根据reducer对in中的数据进行数据规约，当ReduceMode == kLocalMode时，对in沿着NX方向进行规约，完成线程内规约，in数据size为[NY， NX]， dst为[NY, 1]，当ReduceMode == kGlobalMode时,使用共享内存完成block内线程间的规约操作，in和out的size相同，均为[NY,NX]

###模板参数
```
T: 输入数据的类型
NX：有NX列数据参与计算
NY：有NY行数据参与计算
BlockSize：设备属性，标识当前设备线程索引方法。对于GPU，threadIdx.x用作线程索引，而对于xpu，core_id（）用作线程索引。
ReduceFunctor: Reduce计算函数，定义方式如下：
  template <typename InT>
  struct XxxFunctor {
     HOSTDEVICE OutT operator()(const InT& a, const InT& b) const {
       return ...;
     }
  };

Mode: 规约模式，可以取值为kGlobalMode， kLocalMode，当ReduceMode == kLocalMode时，对in沿着NX方向进行规约操作，完成线程内reduce
当ReduceMode == kGlobalMode时,使用共享内存完成block内线程间的规约操作。

```
###函数参数
```
out：out的寄存器指针，大小为NX*NY。
in：in的寄存器指针，大小为NX*NY。
reducer: 规约方式，可以使用ReduceFunctor<InT>()进行定义。
reduce_last_dim: 表示原始输入的最后一维是否参与reduce
```
