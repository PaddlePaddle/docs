# 低精度算子单测开发规范

## 一、FP16单测添加

### Step1：确定任务情况

**任务明细表见附录**

1. **寻找任务**

   在任务总表中查看‘**实际开发者**’字段，寻找到自己的算子/单测开发任务

2. **算子对应单侧文件位置**

   算子对应的单测在目录**python/paddle/fluid/test/unittests**下，每个算子对应的单测文件可参考任务总表中的 ‘**单测文件**’ 字段

3. **算子对应类别**

   算子所属类别可以参考 ‘**分类**’ 字段

4. **判断算子需要添加还是完善单测**

   算子需要完成的任务在 ‘**任务统计**’ 字段中给出，主要分为2个类别：

   1. ‘**增加FP16支持**’字段为‘**是**’。 需要添加FP16算子支持（参考[低精度算子支持开发规范](https://github.com/PaddlePaddle/docs/tree/develop/docs/dev_guides/amp\_precision/amp\_op\_dev\_guide\_cn.md)），同时需要添加FP16的单测支持，单测添加过程参考Step2。
   2. ‘**增加FP16支持**’字段为空，‘**完善FP16单测**’字段为‘**是**’  。需要完善FP16单测，单测相应问题在 **‘单测添加所需注意事项或存在问题’** 中给出，主要分为三类
      1. 补充单测。**参考Step2。**
      2. 修改为和FP32对比。参考**Step2 的2、3、4。**
      3. 修改阈值设置。**参考Step2的3、4。**

### Step2：具体单测添加步骤

#### 1、确定单测类名和单测的基类

1. 对于添加FP16单测，如果原单测文件内部含有OpTest，则添加继承OpTest的FP16的单测。建议采用“Test” + Op_name + “FP16OP”作为FP16单测类名。
2. 对于完善FP16单测，原先的类名不需要修改。

#### 2、添加/修改OpTest

##### 2.1 修改setUp方法

定义完类的头部后，则对内部的方法进行修改。

对于添加FP16单测，参考**a、b、c**完成setUp函数的添加。

对于修改和FP32结果对比，需要修改self.inputs和self.outputs的传入数据，参考**c。**

主要有以下几个修改：

**a**) 修改self.op_type。 **如代码1-1的第5行。** 对于要设置的值可以参考同一个单测文件内部已有的单测类的self.op_type情况。

**b**) 修改self.dtype。设置为np.float16。**如代码1-1的第7行。**

**c**) 修改输入self.inputs和输出self.outputs。

1. 数据生成。**如代码1-1的第9行所示。**

需要生成FP32类型数据。可使用astype(np.float32)转换成FP32格式

生成时通常采用numpy.random包。多数情况下使用numpy.random.random或numpy.random.uniform函数。

具体的函数、shape形状、参考同文件下的其他单测的设置。

2. 设置self.inputs。**如代码1-1的第13行所示。**

inputs部分需要传入FP16格式的数据。可使用astype(self.dtype)完成转换。

3. 设置self.outputs。**如代码1-1的第15行所示。**

首先需要对输入数据进行计算，对于复杂一些的计算，可能会使得setUp函数过分冗长，可以写成额外的函数， **如代码1-1的第13行** 。

outpus部分需要传入FP32格式的参考结果。

**代码1-1**

```python
class TestAFP16OP(OpTest):
    #...
    def setUp(self):
        #self.op_type用来指定当前OP的类型，可参考同单测文件下的op_type
        self.op_type = 'A'
        #dtype需要设置为np.float16形式
        self.dtype = np.float16
        #生成初始输入数据x，通常使用numpy.random包
        x = np.random.rand(2，3，5).astype(np.float32)
        #计算输出数据out
        #复杂的计算可以自己编写函数完成计算
        #简单的计算可以直接在setUp中计算，如加法等
        out,... = self.compute_output(x,...)
        #inputs需要传入FP16类型的数据，用作单测的输入数据
        self.inputs = {'X': x.astype(self.dtype)}
        #outputs需要传入FP32类型的数据，用作单测的参考数据
        self.outputs = {'Out': out}

    #op_type = 'sequence_reshape'的单测中的compute_coutput
    def compute_output(self, x, x_lod, dimension):
        x_width = x.shape[1]
        out_lod = [[]]
        for i in range(len(x_lod[0])):
            seq_len = x_lod[0][i]
            offset = (seq_len * x_width) / dimension
            assert int(offset) * dimension == seq_len * x_width
            out_lod[0].append(int(offset))
        out = np.zeros(shape=(sum(out_lod[0]), dimension)).astype('float64')
        out.ravel()[:] = x.ravel()[:]
        return out, out_lod
```

#####  2.2 修改test_check_output方法

test_check_output中添加对check_output的调用。

在check_output_with_place调用时，按照各分类传入建议的绝对误差阈值atol。

绝对误差阈值应当按照输出结果根据公式估算

$$
（E = 2^{\lfloor\log_{2}^{E_{out}}\rfloor-10}）(公式1-1)
$$

绝大多数单测生成的数据和结果在[0, 2)之间，所以推荐各类别可使用阈值建议如下，如果存在核验不通过可通过公式1-1按照**输出结果的最大取值**计算推荐绝对误差。

1. 不涉及计算。设置阈值1e-3。

2. 基本运算算子。设置阈值1e-3。

3. 基本数学函数，主要为激活函数。设置阈值1e-3。

4. 累加函数。

对于 $E_{out}$, 可以按照算子计算公式、输入数据的数学期望、输入数据个数来进行估计。

i.纯累加。可以用如下估算误差值。

$$
E_{out}=E_{in} * N = \frac{max + min}{2} * N ，[min, max) 随机生成方式
$$

ii.取平均。可以用如下估算误差值。

$$
E_{out}=E_{in} = \frac{max + min}{2}，[min, max) 随机生成方式
$$

iii.Softmax类。设置阈值1e-3

iv.interp类。设置阈值1e-3

v.norm类。设置阈值1e-3

vi.matmul。设置阈值1e-3

**示例**：

我们在代码1-3中以reduce_sum为例。

在setUp中，我们生成了1000大小的累加序列。采用的是[0, 0.1)的均匀分布，数学期望 $E_{in}=0.05$

对于1000个数的累加结果的数学期望为 $E_{out}=E_{in}*1000=50$

那我们的估算绝对误差为  $E=2^{\lfloor\log_{2}^{E_{out}}\rfloor-10}=0.03125$

在check_output时通过atol参数设置该误差

```python
class TestAFP16OP(OpTest):
    #...
    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                #使用atol指定前向计算的绝对误差阈值
                self.check_output_with_place(place, atol=1e-3)
```

```python
class TestSumOpFP16(OpTest):
    #...
    def setUp(self):
        #...
        x = np.random.uniform(0, 0.1, (1000)).astype('float32')
        #...
    def test_check_output(self):
        self.check_output(check_eager=True, atol=0.03125)

```

##### 2.3 修改test_check_grad方法

test_check_grad中添加对check_grad的调用。

在check_grad_with_place调用时，按照各分类传入建议的相对误差阈值max_relative_error。

各类别处理：

1. 不涉及计算。设置阈值1e-3。

2. 基本运算算子。设置阈值1e-3。

3. 基本数学函数，主要为激活函数。设置阈值1e-2, 对于exp, expm1,tan,cosh,sinh,reciprocal,square, Stanh采用阈值0.1。

4. 累加函数。设置阈值5e-3。
   1. norm。设置阈值5e-3
   2. softmax。设置阈值1e-3
   3. interp。设置阈值1e-2
   4. matmul。设置阈值5e-3
   5. cumsum。设置阈值1e-2。
   6. logsum。设置阈值1e-2。
   7. logcumsumexp。设置阈值0.5。

```python
def TestAFP16OP(OpTest):
    #...
    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                #使用max_relative_error指定反响的相对误差阈值
                self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=1e-2)
```

## 二、BF16单测添加

### Step1：确定任务情况

1. 寻找任务

   a. 在任务总表中查看‘实际开发者’字段，寻找到自己的算子/单测开发任务
2. 算子对应单侧文件位置

   a. 算子对应的单测在目录**python/paddle/fluid/test/unittests**下，每个算子对应的单测文件可参考任务总表中的 **“单测文件”** 字段
3. 算子对应类别

   a. 算子所属类别可以参考 **‘分类’** 字段，按照Step2部分的指引添加相应的单测
4. 判断算子需要添加还是完善单测

   a. 算子需要完成的任务在 **‘任务统计’** 字段中给出，主要分为2个类别：

   1. **‘增加BF16支持’** 字段为 **‘是’** 。需要添加BF16算子支持（参考算子添加规范），同时需要添加BF16的单测支持，单测添加过程参考Step2。
   2. **‘增加BF16支持’** 字段为 **空** ， **‘完善BF16单测’** 字段为‘**是**’ 。需要完善BF16单测，单测相应问题在 **‘单测添加所需注意事项或存在问题’** 中给出，主要分为两类
      1. 补充单测
      2. 修改阈值设置

### Step2：具体单测添加步骤

#### 1. 确定单测类名和单测的基类

对于添加BF16单测，建议是 “Test” + Op_name + “BF16”作为BF16单测类名，将OpTest作为基类。**如代码2-1的第1行。**

#### 2. 修改setUp方法

setUp中需要完成数据生成，添加输入和输出数据。setUp需要设置的内容，参考同一个单测文件下已有的单测类。但主要有以下几点。

首先，修改self.op_type。 **如代码2-1的第5行。** 对于要设置的值可以参考同一个单测文件内部已有的单测类的self.op_type情况。

其次，修改self.dtype。设置为np.uint16。**如代码2-1的第7行。**

最后，修改输入self.inputs和输出self.outputs。

BF16在传入输入和输入参考值时需要调用**convert_float_to_uint16**方法。

1. 数据生成。 **如代码2-1的第9行所示** 。

需要生成FP32类型数据。可使用astype(np.float32)转换成FP32格式

生成时通常采用numpy.random包。利用**numpy.random.random**或 **numpy.random.uniform** 。

具体的函数、shape形状、参考同文件下的其他单测的设置。

2. 设置self.inputs。**如代码2-1的第13行所示。**

inputs部分需要传入Uint16格式的数据。可使用**convert_float_to_uint16**完成转换。

3. 设置self.outputs。**如代码2-1的第15行所示。**

outpus部分需要传入Uint16格式的参考结果。可使用**convert_float_to_uint16**完成转换。

**代码2-1**

```python
def TestABF16(OpTest):
    #...
    def setUp(self):
        #self.op_type用来指定当前OP的类型，可参考同单测文件下的op_type
        self.op_type = 'A'
        #dtype需要设置为np.uint16形式
        self.dtype = np.uint16
        #生成初始输入数据x,通常使用numpy.random包
        x = np.random.rand(2，3，5).astype(np.float32)
        #计算输出数据out
        #复杂的计算可以自己编写函数完成计算
        #简单的计算可以直接在setUp中计算，如加法等
        out = compute_out(x)
        #inputs需要传入uint16类型的数据，使用convert_float_to_uint16来获得
        self.inputs = {'X': convert_float_to_uint16(x)}
        #outputs需要传入uint16类型的数据，使用convert_float_to_uint16来获得
        self.outputs = {'Out': convert_float_to_uint16(out)}
```

#### 3. 修改test_check_output方法

test_check_output中添加对check_output的调用。

在check_output_with_place调用时，按照各分类传入建议的绝对误差阈值atol。

绝对误差阈值应当按照输出结果﻿﻿﻿根据公式估算

$$
E = 2^{\lfloor\log_{2}^{E_{out}}\rfloor-8}（公式2-1)
$$

绝大多数单测生成的数据和结果在[0, 2)之间，所以推荐各类别可使用阈值建议如下，如果存在核验不通过可通过公式1-1按照输出结果的最大取值计算推荐绝对误差。

1. 不涉及计算。设置阈值1e-2。

2. 基本运算算子。设置阈值1e-2。

3. 基本数学函数，主要为激活函数。设置阈值1e-2。

4. 累加函数。

i.纯累加。可以用如下估算误差值。

$$
E_{out}=E_{in} * N = \frac{max + min}{2} * N ，[min, max) 随机生成方式
$$

ii.取平均。可以用如下估算误差值。

$$
E_{out}=E_{in} = \frac{max + min}{2}，[min, max) 随机生成方式
$$

iii.Softmax类。设置阈值1e-2

iv.interp类。设置阈值1e-2

v.norm类。设置阈值1e-2

vi.matmul。设置阈值1e-2

```python
def TestABF16(OpTest):
    #...
    def test_check_output(self):
        self.check_output(atol=1e-2)
```

#### 4. 修改test_check_grad方法

test_check_grad中添加对check_grad的调用。

各类别处理：

1. 不涉及计算。设置阈值1e-2。

2. 基本运算算子。设置阈值1e-2。

3. 基本数学函数，主要为激活函数。设置阈值1e-2。

4. 累加函数。设置阈值1e-2。

```python
def TestABF16(OpTest):
    #...
    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=1e-2)
```

## 三、验证单测添加是否正确

### 3.1 本地环境验证

建议优先通过本地环境进行调试。

1. 编译时，请打开测试选项  **-DWITH_TESTING=ON** ，并使用make -j$(nproc)完成Paddle编译

```bash
cmake .. -DWITH_GPU=ON -DWITH_TESTING=ON
make -j $(nproc)
```

2. pip 安装编译好的whl包，位于build/python/dist下。

```bash
pip install python/dist/paddlepaddle-0.0.0-cpXX-cpXXm-linux_x86_64.whl
```

3. 运行单测，验证是否通过。

```bash
# 指定单测
make test ARGS="-R test_mul_op -V"   # test_xxx是单测文件名称
或
ctest -R test_mul_op [-V] #-V可以打印详细的测试信息
```

4. 如果通过，则添加正确；如果没有通过，请根据报错信息完成修改。

### 3.2 CI验证

1. 提交PR以后，CI会对本次修改进行检查，出错的单测将被报出，可在全量日志中这次报错是否与本次修改有关

### 3.3 特定CI的Approve

1. 对于单测精度阈值（atol, max_relative_error等）的修改会出触发CI-Approval，请根据CI报错的指引，请对应的RD review并approve这个PR

## 四、常见问题总结

### 1. op_type、op_name和测试文件对应关系

| op_type（在OpTest中调用的名字）                              | op_name（Paddle中注册的名字）                                | 测试文件                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------- |
| sum                                                          | add_n                                                        | test_sum_op.py                                  |
| gaussian_random                                              | gaussian                                                     | test_gaussian_random_op.py                      |
| elementiwise_add/sub/mul/div/max/min                         | add/substract/multiply/divide/maximum/minimum                | test_elementiwise_add/sub/mul/div/max/min_op.py |
| argmax/argmin                                                | argmax/argmin                                                | test_arg_min_max_op.py                          |
| bilinear_interp                                              | bilinear                                                     | test_bilinear_interp_op.py                      |
| bicubic_interp                                               | bicubic                                                      | test_bicubic_interp_op.py                       |
| check_finite_and_unscale                                     | check_finite_and_unscale                                     | test_amp_check_finite_and_scale_op.py           |
| softmax_with_cross_entropy                                   | cross_entropy_with_softmax                                   | test_softmax_with_cross_entropy_op.py           |
| depthwise_conv2d                                             | depthwise_conv2d                                             | test_conv2d_op_depthwise_conv.py                |
| gaussian                                                     | gaussian_random                                              | test_gaussian_random_op.py                      |
| less_than/less_equal/greater_than/greater_equal/equal/not_equal | less_than/less_equal/greater_than/greater_equal/equal/not_equal | test_compare_op.py                              |
| isinf/isnan                                                  | isinf/isnan                                                  | test_isfinite_op.py                             |
| matmul_v2                                                    | matmul                                                       | test_matmul_v2_op.py                            |
| segment_pool                                                 | segment_pool                                                 | test_segment_ops.py                             |
| tril、triu                                                   | tril_triu                                                    | test_tril_triu_op.py                            |
| uniform_random                                               | uniform                                                      | test_uniform_random_op.py                       |
| fused_softmax_mask                                           | fused_softmax_mask                                           | test_softmax_mask_fuse_op.py                    |
| p_norm                                                       | p_norm                                                       | test_norm_all.py                                |
| reduce_sum                                                   | sum                                                          | test_reduce_op.py                               |
| fill                                                         | fill_any                                                     | test_fill_any_op.py                             |
| range                                                        | arange                                                       | test_arange.op                                  |
| nonzero                                                      | nonzero                                                      | test_nonzero_api.py                             |
| unique                                                       | unique                                                       | test_unique.py                                  |
| linspace                                                     | linspace                                                     | test_linspace.py                                |
