# 运行时设备切换

Paddle提供了[fluid.CUDAPlace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/CUDAPlace_cn.html)以及[fluid.CPUPlace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/CPUPlace_cn.html)用于指定运行时的设备。这两个接口用于指定全局的设备，从1.8版本开始，Paddle提供了[device_guard](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/fluid_cn/device_guard_cn.html)接口，用于指定部分OP的运行设备，此教程会介绍device_guard的使用场景，以及如何使用该接口对模型进行优化。

如果使用了`fluid.CUDAPlace`设置了全局的执行设备，框架将尽可能地将OP设置在GPU上执行，因此有可能会遇到显存不够的情况。`device_guard`可以用于设置OP的执行设备，如果将部分层设置在CPU上运行，就能够充分利用CPU大内存的优势，避免显存超出。

有时尽管指定了全局的执行设备为GPU，但框架在自动分配OP执行设备时，可能会将部分OP设置在CPU上执行。另外，个别OP会将输出存储在CPU上。在以上的场景中，常常会发生不同设备间的数据传输，可能会影响模型的性能。使用`device_guard`可以避免模型运行中不必要的数据传输。在下面的内容中，将会详细介绍如何通过[profile](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_cn/profiler_cn.html)工具分析数据传输开销，以及如何使用`device_guard`避免不必要的数据传输，从而提升模型性能。

## 如何避免显存超出

下面示例代码中的`embedding`层，其参数`size`包含两个元素，第一个元素为`vocab_size` (词表大小), 第二个为`emb_size`（`embedding`层维度）。实际场景中，词表可能会非常大。示例代码中，词表大小被设置为10000000。如果在GPU模式下运行，该层创建的权重矩阵的大小为(10000000, 150)，仅这一层就需要5.59G的显存，如果词表大小继续增加，极有可能会导致显存超出。

```python
import paddle.fluid as fluid

data = fluid.layers.fill_constant(shape=[1], value=128, dtype='int64')
label = fluid.layers.fill_constant(shape=[1, 150], value=0.5, dtype='float32')
emb = fluid.embedding(input=data, size=(10000000, 150), dtype='float32')
out = fluid.layers.l2_normalize(x=emb, axis=-1)

cost = fluid.layers.square_error_cost(input=out, label=label)
avg_cost = fluid.layers.mean(cost)
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_cost)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
result = exe.run(fluid.default_main_program(), fetch_list=[avg_cost])
```

`embedding`是根据`input`中的`id`信息从`embedding`矩阵中查询对应`embedding`信息，在CPU上进行计算，其速度也是可接受的。因此，可以参考如下代码，使用`device_guard`将`embedding`层设置在CPU上，以利用CPU内存资源。那么，除了`embedding`层，其他各层都会在GPU上运行。

```python
import paddle.fluid as fluid

data = fluid.layers.fill_constant(shape=[1], value=128, dtype='int64')
label = fluid.layers.fill_constant(shape=[1, 150], value=0.5, dtype='float32')
with fluid.device_guard("cpu"):
    emb = fluid.embedding(input=data, size=(10000000, 150), dtype='float32')
out = fluid.layers.l2_normalize(x=emb, axis=-1)

cost = fluid.layers.square_error_cost(input=out, label=label)
avg_cost = fluid.layers.mean(cost)
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_cost)

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
result = exe.run(fluid.default_main_program(), fetch_list=[avg_cost])
```

在显存足够的情况下，可不必进行这样的设置。

## 如何减少数据传输
### 使用profile工具确认是否发生了数据传输
首先对模型的性能数据进行分析，找到发生数据传输的原因。如下列代码所示，可以利用[profile](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/profiler_cn.html)工具进行分析。

```python
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.fluid.profiler as profiler

data1 = fluid.layers.fill_constant(shape=[1, 3, 8, 8], value=0.5, dtype='float32')
data2 = fluid.layers.fill_constant(shape=[1, 3, 5, 5], value=0.5, dtype='float32')
shape = fluid.layers.shape(data2)
shape = fluid.layers.slice(shape, axes=[0], starts=[0], ends=[4])
out = fluid.layers.crop_tensor(data1, shape=shape)
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
compiled_prog = compiler.CompiledProgram(fluid.default_main_program())
with profiler.profiler('All', 'total') as prof:
    for i in range(10):
        result = exe.run(program=compiled_prog, fetch_list=[out])
```

在程序运行结束后，将会自动地打印出profile report。在下面的profile report中，可以看到    `GpuMemCpy Summary`中给出了2项数据传输的调用耗时。在OP执行过程中，如果输入Tensor所在的设备与OP执行的设备不同，就会发生`GpuMemcpySync`，通常我们可以直接优化的就是这一项。进一步分析，可以看到`slice`和`crop_tensor`执行中都发生了`GpuMemcpySync`。尽管我们在程序中设置了GPU模式运行，但是框架中有些OP，例如shape，会将输出结果放在CPU上。

```text
------------------------->     Profiling Report     <-------------------------

Note! This Report merge all thread info into one.
Place: All
Time unit: ms
Sorted by total time in descending order in the same thread

Total time: 26.6328
  Computation time       Total: 13.3133     Ratio: 49.9884%
  Framework overhead     Total: 13.3195     Ratio: 50.0116%

-------------------------     GpuMemCpy Summary     -------------------------

GpuMemcpy                Calls: 30          Total: 1.47508     Ratio: 5.5386%
  GpuMemcpyAsync         Calls: 10          Total: 0.443514    Ratio: 1.66529%
  GpuMemcpySync          Calls: 20          Total: 1.03157     Ratio: 3.87331%

-------------------------       Event Summary       -------------------------

Event                                                       Calls       Total       CPU Time (Ratio)        GPU Time (Ratio)        Min.        Max.        Ave.        Ratio.
FastThreadedSSAGraphExecutorPrepare                         10          9.16493     9.152509 (0.998645)     0.012417 (0.001355)     0.025192    8.85968     0.916493    0.344122
shape                                                       10          8.33057     8.330568 (1.000000)     0.000000 (0.000000)     0.030711    7.99849     0.833057    0.312793
fill_constant                                               20          4.06097     4.024522 (0.991025)     0.036449 (0.008975)     0.075087    0.888959    0.203049    0.15248
slice                                                       10          1.78033     1.750439 (0.983212)     0.029888 (0.016788)     0.148503    0.290851    0.178033    0.0668471
  GpuMemcpySync:CPU->GPU                                    10          0.45524     0.446312 (0.980388)     0.008928 (0.019612)     0.039089    0.060694    0.045524    0.0170932
crop_tensor                                                 10          1.67658     1.620542 (0.966578)     0.056034 (0.033422)     0.143906    0.258776    0.167658    0.0629515
  GpuMemcpySync:GPU->CPU                                    10          0.57633     0.552906 (0.959357)     0.023424 (0.040643)     0.050657    0.076322    0.057633    0.0216398
Fetch                                                       10          0.919361    0.895201 (0.973721)     0.024160 (0.026279)     0.082935    0.138122    0.0919361   0.0345199
  GpuMemcpyAsync:GPU->CPU                                   10          0.443514    0.419354 (0.945526)     0.024160 (0.054474)     0.040639    0.059673    0.0443514   0.0166529
ScopeBufferedMonitor::post_local_exec_scopes_process        10          0.341999    0.341999 (1.000000)     0.000000 (0.000000)     0.028436    0.057134    0.0341999   0.0128413
eager_deletion                                              30          0.287236    0.287236 (1.000000)     0.000000 (0.000000)     0.005452    0.022696    0.00957453  0.010785
ScopeBufferedMonitor::pre_local_exec_scopes_process         10          0.047864    0.047864 (1.000000)     0.000000 (0.000000)     0.003668    0.011592    0.0047864   0.00179718
InitLocalVars                                               1           0.022981    0.022981 (1.000000)     0.000000 (0.000000)     0.022981    0.022981    0.022981    0.000862883
```
### 通过log查看发生数据传输的具体位置

以上的示例程序比较简单，我们只用看profile report就能知道具体是哪些算子发生了数据传输。但是当模型比较复杂时，可能需要去查看更加详细的调试信息，可以打印出运行时的log去确定发生数据传输的具体位置。依然以上述程序为例，执行`GLOG_vmodule=operator=3 python test_case.py`，会得到如下log信息，会发现发生了2次数据传输：

- `shape`输出的结果在CPU上，在`slice`运行时，`shape`的输出被拷贝到GPU上
- `slice`执行完的结果在GPU上，当`crop_tensor`执行时，它会被拷贝到CPU上。

```text
I0406 14:56:23.286592 17516 operator.cc:180] CUDAPlace(0) Op(shape), inputs:{Input[fill_constant_1.tmp_0:float[1, 3, 5, 5]({})]}, outputs:{Out[shape_0.tmp_0:int[4]({})]}.
I0406 14:56:23.286628 17516 eager_deletion_op_handle.cc:107] Erase variable fill_constant_1.tmp_0 on CUDAPlace(0)
I0406 14:56:23.286725 17516 operator.cc:1210] Transform Variable shape_0.tmp_0 from data_type[int]:data_layout[NCHW]:place[CPUPlace]:library_type[PLAIN] to data_type[int]:data_layout[ANY_LAYOUT]:place[CUDAPlace(0)]:library_type[PLAIN]
I0406 14:56:23.286763 17516 scope.cc:169] Create variable shape_0.tmp_0
I0406 14:56:23.286784 17516 data_device_transform.cc:21] DeviceTransform in, src_place CPUPlace dst_place: CUDAPlace(0)
I0406 14:56:23.286867 17516 tensor_util.cu:129] TensorCopySync 4 from CPUPlace to CUDAPlace(0)
I0406 14:56:23.287099 17516 operator.cc:180] CUDAPlace(0) Op(slice), inputs:{EndsTensor[], EndsTensorList[], Input[shape_0.tmp_0:int[4]({})], StartsTensor[], StartsTensorList[]}, outputs:{Out[slice_0.tmp_0:int[4]({})]}.
I0406 14:56:23.287140 17516 eager_deletion_op_handle.cc:107] Erase variable shape_0.tmp_0 on CUDAPlace(0)
I0406 14:56:23.287220 17516 tensor_util.cu:129] TensorCopySync 4 from CUDAPlace(0) to CPUPlace
I0406 14:56:23.287473 17516 operator.cc:180] CUDAPlace(0) Op(crop_tensor), inputs:{Offsets[], OffsetsTensor[], Shape[slice_0.tmp_0:int[4]({})], ShapeTensor[], X[fill_constant_0.tmp_0:float[1, 3, 8, 8]({})]}, outputs:{Out[crop_tensor_0.tmp_0:float[1, 3, 5, 5]({})]}.
```

### 使用device_guard避免不必要的数据传输

在上面的例子中，`shape`输出的是一个1-D的Tensor，因此对于`slice`而言计算量很小。这种情况下如果将`slice`设置在CPU上运行，就可以避免2次数据传输。修改后的程序如下：

```python
import paddle.fluid as fluid
import paddle.fluid.compiler as compiler
import paddle.fluid.profiler as profiler

data1 = fluid.layers.fill_constant(shape=[1, 3, 8, 8], value=0.5, dtype='float32')
data2 = fluid.layers.fill_constant(shape=[1, 3, 5, 5], value=0.5, dtype='float32')
shape = fluid.layers.shape(data2)
with fluid.device_guard("cpu"):
    shape = fluid.layers.slice(shape, axes=[0], starts=[0], ends=[4])
out = fluid.layers.crop_tensor(data1, shape=shape)
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
compiled_prog = compiler.CompiledProgram(fluid.default_main_program())
with profiler.profiler('All', 'total') as prof:
    for i in range(10):
        result = exe.run(program=compiled_prog, fetch_list=[out])
```
再次观察profile report中`GpuMemCpy Summary`的内容，可以看到`GpuMemCpySync`已经被消除。在实际的模型中，若`GpuMemCpySync` 调用耗时占比较大，并且可以通过设置`device_guard`避免，那么就能够带来一定的性能提升。

```text
------------------------->     Profiling Report     <-------------------------

Note! This Report merge all thread info into one.
Place: All
Time unit: ms
Sorted by total time in descending order in the same thread

Total time: 14.5345
  Computation time       Total: 4.47587     Ratio: 30.7948%
  Framework overhead     Total: 10.0586     Ratio: 69.2052%

-------------------------     GpuMemCpy Summary     -------------------------

GpuMemcpy                Calls: 10          Total: 0.457033    Ratio: 3.14447%
  GpuMemcpyAsync         Calls: 10          Total: 0.457033    Ratio: 3.14447%

-------------------------       Event Summary       -------------------------

Event                                                       Calls       Total       CPU Time (Ratio)        GPU Time (Ratio)        Min.        Max.        Ave.        Ratio.
FastThreadedSSAGraphExecutorPrepare                         10          7.70113     7.689066 (0.998433)     0.012064 (0.001567)     0.032657    7.39363     0.770113    0.529852
fill_constant                                               20          2.62299     2.587022 (0.986287)     0.035968 (0.013713)     0.071097    0.342082    0.13115     0.180466
shape                                                       10          1.93504     1.935040 (1.000000)     0.000000 (0.000000)     0.026774    1.6016      0.193504    0.133134
Fetch                                                       10          0.880496    0.858512 (0.975032)     0.021984 (0.024968)     0.07392     0.140896    0.0880496   0.0605797
  GpuMemcpyAsync:GPU->CPU                                   10          0.457033    0.435049 (0.951898)     0.021984 (0.048102)     0.037836    0.071424    0.0457033   0.0314447
crop_tensor                                                 10          0.705426    0.671506 (0.951916)     0.033920 (0.048084)     0.05841     0.123901    0.0705426   0.0485346
slice                                                       10          0.324241    0.324241 (1.000000)     0.000000 (0.000000)     0.024299    0.07213     0.0324241   0.0223084
eager_deletion                                              30          0.250524    0.250524 (1.000000)     0.000000 (0.000000)     0.004171    0.016235    0.0083508   0.0172365
ScopeBufferedMonitor::post_local_exec_scopes_process        10          0.047794    0.047794 (1.000000)     0.000000 (0.000000)     0.003344    0.014131    0.0047794   0.00328831
InitLocalVars                                               1           0.034629    0.034629 (1.000000)     0.000000 (0.000000)     0.034629    0.034629    0.034629    0.00238254
ScopeBufferedMonitor::pre_local_exec_scopes_process         10          0.032231    0.032231 (1.000000)     0.000000 (0.000000)     0.002952    0.004076    0.0032231   0.00221755
```

### 总结

- 使用profile工具对模型进行分析，看是否存在GpuMemcpySync的调用耗时。若存在，则进一步分析发生数据传输的原因。
- 可以通过profile report找到发生GpuMemcpySync的OP。如果需要，可以通过打印log，找到GpuMemcpySync发生的具体位置。
- 尝试使用`device_guard`设置部分OP的运行设备，来减少GpuMemcpySync的调用。
- 最后可以通过比较修改前后模型的profile report，或者其他用来衡量性能的指标，确认修改后是否带来了性能提升。
