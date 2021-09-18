##ElementwiseBinary

###函数定义

```
template <typename InT, typename OutT, int NX, int NY, int BlockSize, class OpFunc>
__device__ __forceinline__ void ElementwiseBinary(OutT* out, const InT* in1, const InT* in2, OpFunc compute)；

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
compute：声明为OpFunc<InT，OutT>（）的计算。
```
