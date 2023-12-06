# 低精度算子单测开发规范

## 一、FP16 单测添加

### Step1：确定任务情况

**任务明细表见附录**

1. **寻找任务**

   在任务总表中查看‘**实际开发者**’字段，寻找到自己的算子/单测开发任务

2. **算子对应单侧文件位置**

   算子对应的单测在目录**test/legacy_test**下，每个算子对应的单测文件可参考任务总表中的 ‘**单测文件**’ 字段

3. **算子对应类别**

   算子所属类别可以参考 ‘**分类**’ 字段

4. **判断算子需要添加还是完善单测**

   算子需要完成的任务在 ‘**任务统计**’ 字段中给出，主要分为 2 个类别：

   1. ‘**增加 FP16 支持**’字段为‘**是**’。 需要添加 FP16 算子支持（参考[低精度算子支持开发规范](https://github.com/PaddlePaddle/docs/tree/develop/docs/dev_guides/amp\_precision/amp\_op\_dev\_guide\_cn.md)），同时需要添加 FP16 的单测支持，单测添加过程参考 Step2。
   2. ‘**增加 FP16 支持**’字段为空，‘**完善 FP16 单测**’字段为‘**是**’  。需要完善 FP16 单测，单测相应问题在 **‘单测添加所需注意事项或存在问题’** 中给出，主要分为两类
      1. 补充单测。**参考 Step2。**
      2. 修改阈值设置。**参考 Step2 的 3、4。**

### Step2：具体单测添加步骤

**单测添加原则**：

1. 若有 FP32 类型的 OpTest，则 FP16 的 OpTest 测例数量需与之对应
2. 若无 FP32 类型 OpTest，只有 API Test，则需添加 FP32 及 FP16 的 OpTest，添加过程见 2，添加测例数量与 API Test 对应

#### 1、确定单测类名和单测的基类

1. 对于添加 FP16 单测
   1. 如果原单测文件内部含有 OpTest，则添加继承 OpTest 的 FP16 的单测。建议采用“Test” + Op_name + “FP16OP”作为 FP16 单测类名。
   2. 如果原单测文件内部仅含有 APITest，则需添加 FP32 和 FP16 的 OpTest，命名规则与上述相同。
2. 对于完善 FP16 单测，原先的类名不需要修改。

#### 2、添加/修改 OpTest

##### 2.1 修改 setUp 方法

定义完类的头部后，则对内部的方法进行修改。

对于添加 FP32/FP16 单测，参考以下步骤完成 setUp 函数的添加。

主要有以下几个修改：

**a**) 修改 self.op_type。 **如代码 1-1 的第 5 行。** 对于要设置的值可以参考同一个单测文件内部已有的单测类的 self.op_type 情况。

**b**) 修改 self.dtype。设置为 np.float16。**如代码 1-1 的第 7 行。**

**c**) 修改输入 self.inputs 和输出 self.outputs。

1. 数据生成。**如代码 1-1 的第 9 行所示。**

   需要生成与 self.dtype 类型相同的数据。可使用 astype(self.dtype)完成

   生成时通常采用 numpy.random 包。多数情况下使用 numpy.random.random 或 numpy.random.uniform 函数。

   具体的函数、shape 形状、参考同文件下的其他单测的设置。

2. 设置 self.inputs。**如代码 1-1 的第 15 行所示。**

   inputs 部分需要上述生成的输入数据。

3. 设置 self.outputs。**如代码 1-1 的第 17 行所示。**

   首先需要对输入数据进行计算，对于复杂一些的计算，可能会使得 setUp 函数过分冗长，可以写成额外的函数， **如代码 1-1 的第 13 行** 。

   outpus 部分需要传入由 numpy 计算出的参考结果。

**代码 1-1**

```python
class TestAFP16OP(OpTest):
    #...
    def setUp(self):
        #self.op_type 用来指定当前 OP 的类型，可参考同单测文件下的 op_type
        self.op_type = 'A'
        #dtype 需要设置为 np.float16 形式
        self.dtype = np.float16
        #生成初始输入数据 x，通常使用 numpy.random 包
        x = np.random.rand(2，3，5).astype(np.float32)
        #计算输出数据 out
        #复杂的计算可以自己编写函数完成计算
        #简单的计算可以直接在 setUp 中计算，如加法等
        out,... = self.compute_output(x,...)
        #inputs 需要传入 FP16 类型的数据，用作单测的输入数据
        self.inputs = {'X': x.astype(self.dtype)}
        #outputs 需要传入 FP32 类型的数据，用作单测的参考数据
        self.outputs = {'Out': out}

    #op_type = 'sequence_reshape'的单测中的 compute_coutput
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

#####  2.2 修改 test_check_output 方法

test_check_output 中添加对 check_output 的调用，如代码 1-2 所示。

在 check_output_with_place 调用时，按照各分类传入建议的绝对误差阈值 atol。

绝对误差阈值应当按照输出结果根据公式估算

$$
（E = 2^{\lfloor\log_{2}^{E_{out}}\rfloor-10}）(公式 1-1)
$$

绝大多数单测生成的数据和结果在[0, 2)之间，所以推荐各类别可使用阈值建议如下，如果存在核验不通过可通过公式 1-1 按照**输出结果的最大取值**计算推荐绝对误差。

1. 不涉及计算。设置阈值 1e-3。

2. 基本运算算子。设置阈值 1e-3。

3. 基本数学函数，主要为激活函数。设置阈值 1e-3。

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

iii.Softmax 类。设置阈值 1e-3

iv.interp 类。设置阈值 1e-3

v.norm 类。设置阈值 1e-3

vi.matmul。设置阈值 1e-3

**示例**：

我们在代码 1-3 中以 reduce_sum 为例。

在 setUp 中，我们生成了 1000 大小的累加序列。采用的是[0, 0.1)的均匀分布，数学期望 $E_{in}=0.05$

对于 1000 个数的累加结果的数学期望为 $E_{out}=E_{in}*1000=50$

那我们的估算绝对误差为  $E=2^{\lfloor\log_{2}^{E_{out}}\rfloor-10}=0.03125$

在 check_output 时通过 atol 参数设置该误差

**代码 1-2**

```python
class TestAFP16OP(OpTest):
    #...
    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                #使用 atol 指定前向计算的绝对误差阈值
                self.check_output_with_place(place, atol=1e-3)
```

**代码 1-3**

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

##### 2.3 修改 test_check_grad 方法

test_check_grad 中添加对 check_grad 的调用，如代码 1-4 所示。

在 check_grad_with_place 调用时，按照各分类传入建议的相对误差阈值 max_relative_error。

各类别处理：

1. 不涉及计算。设置阈值 1e-3。

2. 基本运算算子。设置阈值 1e-3。

3. 基本数学函数，主要为激活函数。设置阈值 1e-2, 对于 exp, expm1,tan,cosh,sinh,reciprocal,square, Stanh 采用阈值 0.1。

4. 累加函数。设置阈值 5e-3。
   1. norm。设置阈值 5e-3
   2. softmax。设置阈值 1e-3
   3. interp。设置阈值 1e-2
   4. matmul。设置阈值 5e-3
   5. cumsum。设置阈值 1e-2。
   6. logsum。设置阈值 1e-2。
   7. logcumsumexp。设置阈值 0.5。

**代码 1-4**

```python
def TestAFP16OP(OpTest):
    #...
    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                #使用 max_relative_error 指定反响的相对误差阈值
                self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=1e-2)
```

## 二、BF16 单测添加

### Step1：确定任务情况

1. 寻找任务

   a. 在任务总表中查看‘实际开发者’字段，寻找到自己的算子/单测开发任务
2. 算子对应单侧文件位置

   a. 算子对应的单测在目录**test/legacy_test**下，每个算子对应的单测文件可参考任务总表中的 **“单测文件”** 字段
3. 算子对应类别

   a. 算子所属类别可以参考 **‘分类’** 字段，按照 Step2 部分的指引添加相应的单测
4. 判断算子需要添加还是完善单测

   a. 算子需要完成的任务在 **‘任务统计’** 字段中给出，主要分为 2 个类别：

   1. **‘增加 BF16 支持’** 字段为 **‘是’** 。需要添加 BF16 算子支持（参考算子添加规范），同时需要添加 BF16 的单测支持，单测添加过程参考 Step2。
   2. **‘增加 BF16 支持’** 字段为 **空** ， **‘完善 BF16 单测’** 字段为‘**是**’ 。需要完善 BF16 单测，单测相应问题在 **‘单测添加所需注意事项或存在问题’** 中给出，主要分为两类
      1. 补充单测
      2. 修改阈值设置

### Step2：具体单测添加步骤

#### 1. 确定单测类名和单测的基类

对于添加 BF16 单测，建议是 “Test” + Op_name + “BF16”作为 BF16 单测类名，将 OpTest 作为基类。**如代码 2-1 的第 1 行。**

#### 2. 修改 setUp 方法

setUp 中需要完成数据生成，添加输入和输出数据。setUp 需要设置的内容，参考同一个单测文件下已有的单测类。但主要有以下几点。

首先，修改 self.op_type。 **如代码 2-1 的第 5 行。** 对于要设置的值可以参考同一个单测文件内部已有的单测类的 self.op_type 情况。

其次，修改 self.dtype。设置为 np.uint16。**如代码 2-1 的第 7 行。**

最后，修改输入 self.inputs 和输出 self.outputs。

BF16 在传入输入和输入参考值时需要调用**convert_float_to_uint16**方法。

1. 数据生成。 **如代码 2-1 的第 9 行所示** 。

   需要生成 FP32 类型数据。可使用 astype(np.float32)转换成 FP32 格式

   生成时通常采用 numpy.random 包。利用**numpy.random.random**或 **numpy.random.uniform** 。

   具体的函数、shape 形状、参考同文件下的其他单测的设置。

2. 设置 self.inputs。**如代码 2-1 的第 13 行所示。**

   inputs 部分需要传入 Uint16 格式的数据。可使用**convert_float_to_uint16**完成转换。

3. 设置 self.outputs。**如代码 2-1 的第 15 行所示。**

   outpus 部分需要传入 Uint16 格式的参考结果。可使用**convert_float_to_uint16**完成转换。

**代码 2-1**

```python
def TestABF16(OpTest):
    #...
    def setUp(self):
        #self.op_type 用来指定当前 OP 的类型，可参考同单测文件下的 op_type
        self.op_type = 'A'
        #dtype 需要设置为 np.uint16 形式
        self.dtype = np.uint16
        #生成初始输入数据 x,通常使用 numpy.random 包
        x = np.random.rand(2，3，5).astype(np.float32)
        #计算输出数据 out
        #复杂的计算可以自己编写函数完成计算
        #简单的计算可以直接在 setUp 中计算，如加法等
        out = compute_out(x)
        #inputs 需要传入 uint16 类型的数据，使用 convert_float_to_uint16 来获得
        self.inputs = {'X': convert_float_to_uint16(x)}
        #outputs 需要传入 uint16 类型的数据，使用 convert_float_to_uint16 来获得
        self.outputs = {'Out': convert_float_to_uint16(out)}
```

#### 3. 修改 test_check_output 方法

test_check_output 中添加对 check_output 的调用，如代码 2-2 所示。

在 check_output_with_place 调用时，按照各分类传入建议的绝对误差阈值 atol。

绝对误差阈值应当按照输出结果﻿﻿﻿根据公式估算

$$
E = 2^{\lfloor\log_{2}^{E_{out}}\rfloor-8}（公式 2-1)
$$

绝大多数单测生成的数据和结果在[0, 2)之间，所以推荐各类别可使用阈值建议如下，如果存在核验不通过可通过公式 1-1 按照输出结果的最大取值计算推荐绝对误差。

1. 不涉及计算。设置阈值 1e-2。

2. 基本运算算子。设置阈值 1e-2。

3. 基本数学函数，主要为激活函数。设置阈值 1e-2。

4. 累加函数。

i.纯累加。可以用如下估算误差值。

$$
E_{out}=E_{in} * N = \frac{max + min}{2} * N ，[min, max) 随机生成方式
$$

ii.取平均。可以用如下估算误差值。

$$
E_{out}=E_{in} = \frac{max + min}{2}，[min, max) 随机生成方式
$$

iii.Softmax 类。设置阈值 1e-2

iv.interp 类。设置阈值 1e-2

v.norm 类。设置阈值 1e-2

vi.matmul。设置阈值 1e-2

**代码 2-2**

```python
def TestABF16(OpTest):
    #...
    def test_check_output(self):
        self.check_output(atol=1e-2)
```

#### 4. 修改 test_check_grad 方法

test_check_grad 中添加对 check_grad 的调用，如代码 2-3 所示。

各类别处理：

1. 不涉及计算。设置阈值 1e-2。

2. 基本运算算子。设置阈值 1e-2。

3. 基本数学函数，主要为激活函数。设置阈值 1e-2。

4. 累加函数。设置阈值 1e-2。

**代码 2-3**

```python
def TestABF16(OpTest):
    #...
    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=1e-2)
```

## 三、验证单测添加是否正确

### 3.1 本地环境验证

建议优先通过本地环境进行调试。

1. 编译时，请打开测试选项  **-DWITH_TESTING=ON** ，并使用 make -j$(nproc)完成 Paddle 编译

   ```bash
   cmake .. -DWITH_GPU=ON -DWITH_TESTING=ON
   make -j $(nproc)
   ```

2. pip 安装编译好的 whl 包，位于 build/python/dist 下。

   ```bash
   pip install python/dist/paddlepaddle-0.0.0-cpXX-cpXXm-linux_x86_64.whl
   ```

3. 运行单测，验证是否通过。

   ```bash
   # 指定单测
   make test ARGS="-R test_mul_op -V"   # test_xxx 是单测文件名称
   或
   ctest -R test_mul_op [-V] #-V 可以打印详细的测试信息
   ```

4. 如果通过，则添加正确；如果没有通过，请根据报错信息完成修改。

### 3.2 CI 验证

1. 提交 PR 以后，CI 会对本次修改进行检查，出错的单测将被报出，可在全量日志中这次报错是否与本次修改有关

### 3.3 特定 CI 的 Approve

1. 对于单测精度阈值（atol, max_relative_error 等）的修改会出触发 CI-Approval，请根据 CI 报错的指引，请对应的 RD review 并 approve 这个 PR

## 四、常见问题总结

### 1. op_type、op_name 和测试文件对应关系

| op_type（在 OpTest 中调用的名字）                              | op_name（Paddle 中注册的名字）                                | 测试文件                                        |
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
