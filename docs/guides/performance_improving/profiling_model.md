# 模型性能分析
Paddle Profiler是Paddle框架自带的低开销性能分析器，可以对模型运行过程的性能数据进行收集、统计和展示。性能分析器提供的数据可以帮助我们定位模型的瓶颈，识别造成程序运行时间过长或者GPU利用率低的原因，从而寻求优化方案来获得性能的提升。

在这篇文档中，我们主要介绍如何使用Profiler工具来调试程序性能，以及阐述当前提供的所有功能特性。

## 内容

- [使用Profiler工具调试程序性能](#profiler)
- [功能特性](#gongnengtexing)
- [更多细节](#gengduoxijie)


## 使用Profiler工具调试程序性能
在模型性能分析中，通常采用如下四个步骤：
- 获取模型正常运行时的ips(iterations per second, 每秒的迭代次数)，给出baseline数据。
- 开启性能分析器，定位性能瓶颈点。
- 优化程序，检查优化效果。
- 获取优化后模型正常运行时的ips，和baseline比较，确定真实的提升幅度。

我们以一个比较简单的示例，来看性能分析工具是如何通过上述四个步骤在调试程序性能中发挥作用。下面是Paddle的应用实践教学中关于[使用神经网络对cifar10进行分类](https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/cv/convnet_image_classification.html)的示例代码，我们加上了启动性能分析的代码。

```python
def train(model):
    print('start training ... ')
    # turn into training mode
    model.train()

    opt = paddle.optimizer.Adam(learning_rate=learning_rate,
                                parameters=model.parameters())

    train_loader = paddle.io.DataLoader(cifar10_train,
                                        shuffle=True,
                                        batch_size=batch_size)

    valid_loader = paddle.io.DataLoader(cifar10_test, batch_size=batch_size)

    # 创建性能分析器相关的代码
    def my_on_trace_ready(prof):
      callback = profiler.export_chrome_tracing('./profiler_demo')
      callback(prof)
      prof.summary(sorted_by=profiler.SortedKeys.GPUTotal)
    p = profiler.Profiler(scheduler = [3,14], on_trace_ready=my_on_trace_ready, timer_only=True)
    p.start()
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)

            if batch_id % 1000 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
            loss.backward()
            opt.step()
            opt.clear_grad()
            p.step()
            if batch_id == 19:
              p.stop()
              exit() # 做性能分析时，可以将程序提前退出

        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []
        for batch_id, data in enumerate(valid_loader()):
            x_data = data[0]
            y_data = paddle.to_tensor(data[1])
            y_data = paddle.unsqueeze(y_data, 1)

            logits = model(x_data)
            loss = F.cross_entropy(logits, y_data)
            acc = paddle.metric.accuracy(logits, y_data)
            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
        print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
        val_acc_history.append(avg_acc)
        val_loss_history.append(avg_loss)
        model.train()
```


### 1. 获取性能调试前模型正常运行的ips

上述程序在创建Profiler时候，timer_only设置的值为True，此时将只开启benchmark功能，不开启性能分析器，程序输出模型正常运行时的benchmark信息如下
```text
============================================Perf Summary============================================
Reader Ratio: 53.514%
Time Unit: s, IPS Unit: steps/s
|                 |       avg       |       max       |       min       |
|   reader_cost   |     0.01367     |     0.01407     |     0.01310     |
|    batch_cost   |     0.02555     |     0.02381     |     0.02220     |
|       ips       |     39.13907    |     45.03588    |     41.99930    |
```
可以看到，此时的ips为39.1，可将这个值作为优化对比的baseline。


### 2. 开启性能分析器，定位性能瓶颈点

修改程序，将Profiler的timer_only参数设置为False, 此时代表不只开启benchmark功能，还将开启性能分析器，进行详细的性能分析。
```python
p = profiler.Profiler(scheduler = [3,14], on_trace_ready=my_on_trace_ready, timer_only=False)
```

性能分析器会收集程序在第3到14次（不包括14）训练迭代过程中的性能数据，并在profiler_demo文件夹中输出一个json格式的文件，用于展示程序执行过程的timeline，可通过chrome浏览器的[chrome://tracing](chrome://tracing)插件打开这个文件进行观察。
<p align="center">
<img src="https://user-images.githubusercontent.com/22424850/165498308-734b4978-252e-45fc-8376-aaf8eb8a4270.png"   width='80%' hspace='10'/>
<br />
</p>

性能分析器还会直接在终端打印统计表单（建议重定向到文件中查看），查看程序输出的Model Summary表单

```text
-----------------------------------------------Model Summary-----------------------------------------------
Time unit: ms
---------------  ------  ----------------------------------------  ----------------------------------------  
Name             Calls   CPU Total / Avg / Max / Min / Ratio(%)    GPU Total / Avg / Max / Min / Ratio(%)  
---------------  ------  ----------------------------------------  ----------------------------------------  
ProfileStep      11      294.53 / 26.78 / 35.28 / 24.56 / 100.00   13.22 / 1.20 / 1.20 / 1.20 / 100.00  
  Dataloader     11      141.49 / 12.86 / 17.53 / 10.34 / 48.04    0.00 / 0.00 / 0.00 / 0.00 / 0.00  
  Forward        11      51.41 / 4.67 / 6.18 / 3.93 / 17.45        3.92 / 0.36 / 0.36 / 0.35 / 29.50  
  Backward       11      21.23 / 1.93 / 2.61 / 1.70 / 7.21         8.14 / 0.74 / 0.74 / 0.74 / 61.51  
  Optimization   11      34.74 / 3.16 / 3.65 / 2.41 / 11.79        0.67 / 0.06 / 0.06 / 0.06 / 5.03  
  Others         -       45.66 / - / - / - / 15.50                 0.53 / - / - / - / 3.96  
---------------  ------  ----------------------------------------  ----------------------------------------  
```

通过timeline可以看到，dataloader占了执行过程的很大比重，Model Summary显示其甚至接近了50%。分析程序发现，这是由于模型本身比较简单，需要的计算量小，再加上dataloader
准备数据时只用了单进程来读取，使得程序读取数据时和执行计算时没有并行操作，导致dataloader占比过大。

### 3. 优化程序，检查优化效果

识别到了问题产生的原因，我们对程序继续做如下修改，将dataloader的num_workers设置为4，使得能有多个进程并行读取数据。
```python
train_loader = paddle.io.DataLoader(cifar10_train,
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=4)
```

重新对程序进行性能分析，新的timeline和Model Summary如下所示
<p align="center">
<img src="https://user-images.githubusercontent.com/22424850/165498358-100b7e73-de25-47df-9b5d-5b10c887bcbd.png"   width='80%' hspace='10'/>
<br />
</p>

```text
-----------------------------------------------Model Summary-----------------------------------------------
Time unit: ms
---------------  ------  ----------------------------------------  ----------------------------------------  
Name             Calls   CPU Total / Avg / Max / Min / Ratio(%)    GPU Total / Avg / Max / Min / Ratio(%)  
---------------  ------  ----------------------------------------  ----------------------------------------  
ProfileStep      11      90.94 / 8.27 / 11.82 / 7.85 / 100.00      13.27 / 1.21 / 1.22 / 1.19 / 100.00  
  Dataloader     11      1.82 / 0.17 / 0.67 / 0.11 / 2.00          0.00 / 0.00 / 0.00 / 0.00 / 0.00  
  Forward        11      29.58 / 2.69 / 3.53 / 2.52 / 32.52        3.82 / 0.35 / 0.35 / 0.34 / 30.67  
  Backward       11      15.21 / 1.38 / 1.95 / 1.31 / 16.72        8.30 / 0.75 / 0.77 / 0.74 / 60.71  
  Optimization   11      17.55 / 1.60 / 1.92 / 1.55 / 19.30        0.66 / 0.06 / 0.06 / 0.06 / 4.82  
  Others         -       26.79 / - / - / - / 29.46                 0.52 / - / - / - / 3.80  
---------------  ------  ----------------------------------------  ----------------------------------------  
```
可以看到，从dataloader中取数据的时间大大减少，变成了平均只占一个step的2%，并且平均一个step所需要的时间也相应减少了。

### 4. 获取优化后模型正常运行的ips，确定真实提升幅度
重新将timer_only设置的值为True，获取优化后模型正常运行时的benchmark信息

```text
============================================Perf Summary============================================
Reader Ratio: 1.653%
Time Unit: s, IPS Unit: steps/s
|                 |       avg       |       max       |       min       |
|   reader_cost   |     0.00011     |     0.00024     |     0.00009     |
|    batch_cost   |     0.00660     |     0.00629     |     0.00587     |
|       ips       |    151.45498    |    170.28927    |    159.06308    |
```
此时ips的值变成了151.5，相比优化前的baseline 39.1，模型真实性能提升了287%。

**Note** 由于Profiler开启的时候，收集性能数据本身也会造成程序性能的开销，因此正常跑程序时请不要开启性能分析器，性能分析器只作为调试程序性能时使用。如果想获得程序正常运行时候的
benchmark信息（如ips），可以像示例一样将Profiler的timer_only参数设置为True，此时不会进行详尽的性能数据收集，几乎不影响程序正常运行的性能，所获得的benchmark信息也趋于真实。
此外，benchmark信息计算的数据范围是从调用Profiler的start方法开始，到调用stop方法结束这个过程的数据。而Timeline和性能数据的统计表单的数据范围是所指定的采集区间，如这个例子中的第3到14次迭代，这会导致开启性能分析器时统计表单和benchmark信息输出的值不同（如统计到的dataloader的时间占比）。此外，当benchmark统计的范围和性能分析器统计的范围不同时，
由于benchmark统计的是平均时间，如果benchmark统计的范围覆盖了性能分析器开启的范围，也覆盖了关闭性能调试时的正常执行的范围，此时benchmark的值没有意义，因此**开启性能分析器时请以性能分析器输出的统计表单为参考**，这也是为何上面示例里在开启性能分析器时没贴benchmark信息的原因。

## 功能特性

当前Profiler提供Timeline、统计表单、benchmark信息共三个方面的展示功能。

### Timeline展示
对于采集的性能数据，导出为chrome tracing timeline格式的文件后，可以进行可视化分析。当前，所采用的可视化工具为chrome浏览器里的[tracing插件](chrome://tracing)，可以按照如下方式进行查看
  <p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/165717586-599a08fb-c915-4e3c-af40-0732c30c5855.gif"   width='80%' hspace='10'/>
  <br />
  Timeline使用Demo
  </p>
目前Timeline提供以下特性：

1. 查看CPU和GPU在不同线程或stream下的事件发生的时间线。将同一线程下所记录的数据分为Python层和C++层，以便根据需要进行折叠和展开。对于有名字的线程，标注线程名字。
2. 所展示的事件名字上标注事件所持续的时间，点击具体的事件，可在下方的说明栏中看到更详细的事件信息。通过按键'w', 's'可进行放大和缩小，通过'a','d'可进行左移和右移。
3. 对于GPU上的事件，可以通过点击下方的'launch'链接查看所发起它的CPU上的事件。



### 统计表单展示
统计表单负责对采集到的数据(Event)从多个不同的角度进行解读，也可以理解为对timeline进行一些量化的指标计算。
目前提供Device Summary、Overview Summary、Model Summary、Distributed Summary、Operator Summary、Kernel Summary、Memory Manipulation Summary和UserDefined Summary的统计表单，每个表单从不同的角度进行统计计算。每个表单的统计内容简要叙述如下：

- Device Summary
  ```text
  -------------------Device Summary-------------------
  ------------------------------  --------------------  
  Device                          Utilization (%)  
  ------------------------------  --------------------  
  CPU(Process)                    77.13  
  CPU(System)                     25.99  
  GPU2                            55.50  
  ------------------------------  --------------------  
  Note:
  CPU(Process) Utilization = Current process CPU time over all cpu cores / elapsed time, so max utilization can be reached 100% * number of cpu cores.
  CPU(System) Utilization = All processes CPU time over all cpu cores(busy time) / (busy time + idle time).
  GPU Utilization = Current process GPU time / elapsed time.
  ----------------------------------------------------
  ```

  DeviceSummary提供CPU和GPU的平均利用率信息。其中
  - CPU(Process): 指的是进程的cpu平均利用率，算的是从Profiler开始记录数据到结束这一段过程，进程所利用到的 **cpu core的总时间**与**该段时间**的占比。因此如果是多核的情况，对于进程来说cpu平均利用率是有可能超过100%的，因为同时用到的多个core的时间进行了累加。
  - CPU(System): 指的是整个系统的cpu平均利用率，算的是从Profiler开始记录数据到结束这一段过程，整个系统所有进程利用到的**cpu core总时间**与**该段时间乘以cpu core的数量**的占比。可以当成是从cpu的视角来算的利用率。
  - GPU: 指的是进程的gpu平均利用率，算的是从Profiler开始记录数据到结束这一段过程，进程在gpu上所调用的**kernel的执行时间** 与 **该段时间** 的占比。


- Overview Summary

  ```text
  ---------------------------------------------Overview Summary---------------------------------------------
  Time unit: ms
  -------------------------  -------------------------  -------------------------  -------------------------  
  Event Type                 Calls                      CPU Time                   Ratio (%)  
  -------------------------  -------------------------  -------------------------  -------------------------  
  ProfileStep                8                          4945.15                    100.00  
    CudaRuntime              28336                      2435.63                    49.25  
    UserDefined              486                        2280.54                    46.12  
    Dataloader               8                          1819.15                    36.79  
    Forward                  8                          1282.64                    25.94  
    Operator                 8056                       1244.41                    25.16  
    OperatorInner            21880                      374.18                     7.57  
    Backward                 8                          160.43                     3.24  
    Optimization             8                          102.34                     2.07  
  -------------------------  -------------------------  -------------------------  -------------------------  
                            Calls                      GPU Time                   Ratio (%)  
  -------------------------  -------------------------  -------------------------  -------------------------  
    Kernel                   13688                      2744.61                    55.50  
    Memcpy                   496                        29.82                      0.60  
    Memset                   104                        0.12                       0.00  
    Communication            784                        257.23                     5.20  
  -------------------------  -------------------------  -------------------------  -------------------------  
  Note:
  In this table, We sum up all collected events in terms of event type.
  The time of events collected on host are presented as CPU Time, and as GPU Time if on device.
  Events with different types may overlap or inclusion, e.g. Operator includes OperatorInner, so the sum of ratios is not 100%.
  The time of events in the same type with overlap will not calculate twice, and all time is summed after merged.
  Example:
  Thread 1:
    Operator: |___________|     |__________|
  Thread 2:
    Operator:   |____________|     |___|
  After merged:
    Result:   |______________|  |__________|

  ----------------------------------------------------------------------------------------------------------
  ```
  Overview Summary用于展示每种类型的Event一共分别消耗了多少时间，对于多线程或多stream下，如果同一类型的Event有重叠的时间段，我们采取取并集操作，不对重叠的时间进行重复计算。


- Model Summary
  ```text
  --------------------------------------------------Model Summary--------------------------------------------------
  Time unit: ms
  ---------------  ------  -------------------------------------------  -------------------------------------------  
  Name             Calls   CPU Total / Avg / Max / Min / Ratio(%)       GPU Total / Avg / Max / Min / Ratio(%)  
  ---------------  ------  -------------------------------------------  -------------------------------------------  
  ProfileStep      8       4945.15 / 618.14 / 839.15 / 386.34 / 100.00  2790.80 / 348.85 / 372.39 / 344.60 / 100.00  
    Dataloader     8       1819.15 / 227.39 / 451.69 / 0.32 / 36.79     0.00 / 0.00 / 0.00 / 0.00 / 0.00  
    Forward        8       1282.64 / 160.33 / 161.49 / 159.19 / 25.94   1007.64 / 125.96 / 126.13 / 125.58 / 35.90  
    Backward       8       160.43 / 20.05 / 21.00 / 19.21 / 3.24        1762.11 / 220.26 / 243.83 / 216.05 / 62.49  
    Optimization   8       102.34 / 12.79 / 13.42 / 12.47 / 2.07        17.03 / 2.13 / 2.13 / 2.13 / 0.60  
    Others         -       1580.59 / - / - / - / 31.96                  28.22 / - / - / - / 1.00  
  ---------------  ------  -------------------------------------------  -------------------------------------------  
  ```

  Model Summary用于展示模型训练或者推理过程中，dataloader、forward、backward、optimization所消耗的时间。其中GPU Time对应着在该段过程内所发起的GPU侧活动的时间。



- Distributed Summary
  ```text
  -----------------------------Distribution Summary------------------------------
  Time unit: ms
  -------------------------  -------------------------  -------------------------  
  Name                       Total Time                 Ratio (%)  
  -------------------------  -------------------------  -------------------------  
  ProfileStep                4945.15                    100.00  
    Communication            257.23                     5.20  
    Computation              2526.52                    51.09  
    Overlap                  39.13                      0.79  
  -------------------------  -------------------------  -------------------------  
  ```

  Distribution Summary用于展示分布式训练中通信(Communication)、计算(Computation)以及这两者Overlap的时间。

  Communication: 所有和通信有关活动的时间，包括和分布式相关的算子(op)以及gpu上的kernel的时间等。

  Computation: 即是所有kernel在GPU上的执行时间, 但是去除了和通信相关的kernel的时间。

  Overlap: Communication和Computation的重叠时间

- Operator Summary
  ```text
  (由于原始表单较长，这里截取一部分进行展示)
  ----------------------------------------------------------------Operator Summary----------------------------------------------------------------
  Time unit: ms
  ----------------------------------------------------  ------  ----------------------------------------  ----------------------------------------  
  Name                                                  Calls   CPU Total / Avg / Max / Min / Ratio(%)    GPU Total / Avg / Max / Min / Ratio(%)  
  ----------------------------------------------------  ------  ----------------------------------------  ----------------------------------------  
  -----------------------------------------------------------Thread: All threads merged-----------------------------------------------------------
  conv2d_grad grad_node                                 296     53.70 / 0.18 / 0.40 / 0.14 / 4.34         679.11 / 2.29 / 5.75 / 0.24 / 24.11  
    conv2d_grad::infer_shape                            296     0.44 / 0.00 / 0.00 / 0.00 / 0.81          0.00 / 0.00 / 0.00 / 0.00 / 0.00  
    conv2d_grad::compute                                296     44.09 / 0.15 / 0.31 / 0.10 / 82.10        644.39 / 2.18 / 5.75 / 0.24 / 94.89  
      cudnn::maxwell::gemm::computeWgradOffsetsKern...  224     - / - / - / - / -                         0.50 / 0.00 / 0.00 / 0.00 / 0.08  
      void scalePackedTensor_kernel<float, float>(c...  224     - / - / - / - / -                         0.79 / 0.00 / 0.01 / 0.00 / 0.12  
      cudnn::maxwell::gemm::computeBOffsetsKernel(c...  464     - / - / - / - / -                         0.95 / 0.00 / 0.01 / 0.00 / 0.15  
      maxwell_scudnn_128x32_stridedB_splitK_large_nn    8       - / - / - / - / -                         15.70 / 1.96 / 1.97 / 1.96 / 2.44  
      cudnn::maxwell::gemm::computeOffsetsKernel(cu...  240     - / - / - / - / -                         0.54 / 0.00 / 0.00 / 0.00 / 0.08  
      maxwell_scudnn_128x32_stridedB_interior_nn        8       - / - / - / - / -                         9.53 / 1.19 / 1.19 / 1.19 / 1.48  
      maxwell_scudnn_128x64_stridedB_splitK_interio...  8       - / - / - / - / -                         28.67 / 3.58 / 3.59 / 3.58 / 4.45  
      maxwell_scudnn_128x64_stridedB_interior_nn        8       - / - / - / - / -                         5.53 / 0.69 / 0.70 / 0.69 / 0.86  
      maxwell_scudnn_128x128_stridedB_splitK_interi...  184     - / - / - / - / -                         167.03 / 0.91 / 2.28 / 0.19 / 25.92  
      maxwell_scudnn_128x128_stridedB_interior_nn       200     - / - / - / - / -                         105.10 / 0.53 / 0.97 / 0.09 / 16.31  
      MEMSET                                            104     - / - / - / - / -                         0.12 / 0.00 / 0.00 / 0.00 / 0.02  
      maxwell_scudnn_128x128_stridedB_small_nn          24      - / - / - / - / -                         87.58 / 3.65 / 4.00 / 3.53 / 13.59  
      void cudnn::winograd_nonfused::winogradWgradD...  72      - / - / - / - / -                         15.66 / 0.22 / 0.36 / 0.09 / 2.43  
      void cudnn::winograd_nonfused::winogradWgradD...  72      - / - / - / - / -                         31.64 / 0.44 / 0.75 / 0.19 / 4.91  
      maxwell_sgemm_128x64_nt                           72      - / - / - / - / -                         62.03 / 0.86 / 1.09 / 0.75 / 9.63  
      void cudnn::winograd_nonfused::winogradWgradO...  72      - / - / - / - / -                         14.45 / 0.20 / 0.49 / 0.04 / 2.24  
      void cudnn::winograd::generateWinogradTilesKe...  48      - / - / - / - / -                         1.78 / 0.04 / 0.06 / 0.02 / 0.28  
      maxwell_scudnn_winograd_128x128_ldg1_ldg4_til...  24      - / - / - / - / -                         45.94 / 1.91 / 1.93 / 1.90 / 7.13  
      maxwell_scudnn_winograd_128x128_ldg1_ldg4_til...  24      - / - / - / - / -                         40.93 / 1.71 / 1.72 / 1.69 / 6.35  
      maxwell_scudnn_128x32_stridedB_splitK_interio...  24      - / - / - / - / -                         9.91 / 0.41 / 0.77 / 0.15 / 1.54  
    GpuMemcpyAsync:CPU->GPU                             64      0.68 / 0.01 / 0.02 / 0.01 / 1.27          0.09 / 0.00 / 0.00 / 0.00 / 0.01  
      MEMCPY_HtoD                                       64      - / - / - / - / -                         0.09 / 0.00 / 0.00 / 0.00 / 100.00  
    void phi::funcs::ConcatKernel_<float>(float con...  16      - / - / - / - / -                         2.84 / 0.18 / 0.36 / 0.06 / 0.42  
    void phi::funcs::ForRangeElemwiseOp<paddle::imp...  16      - / - / - / - / -                         1.33 / 0.08 / 0.16 / 0.01 / 0.20  
    ncclAllReduceRingLLKernel_sum_f32(ncclColl)         16      - / - / - / - / -                         26.35 / 1.65 / 3.14 / 0.20 / 3.88  
    void phi::funcs::SplitKernel_<float>(float cons...  16      - / - / - / - / -                         2.49 / 0.16 / 0.37 / 0.06 / 0.37  
    void axpy_kernel_val<float, float>(cublasAxpyPa...  16      - / - / - / - / -                         1.63 / 0.10 / 0.14 / 0.07 / 0.24  
  sync_batch_norm_grad grad_node                        376     37.90 / 0.10 / 0.31 / 0.08 / 3.07         670.62 / 1.78 / 39.29 / 0.13 / 23.81  
    sync_batch_norm_grad::infer_shape                   376     1.60 / 0.00 / 0.01 / 0.00 / 4.22          0.00 / 0.00 / 0.00 / 0.00 / 0.00  
    sync_batch_norm_grad::compute                       376     23.26 / 0.06 / 0.10 / 0.06 / 61.37        555.96 / 1.48 / 39.29 / 0.13 / 82.90  
      void paddle::operators::KeBackwardLocalStats<...  376     - / - / - / - / -                         129.62 / 0.34 / 1.83 / 0.04 / 23.32  
      ncclAllReduceRingLLKernel_sum_f32(ncclColl)       376     - / - / - / - / -                         128.00 / 0.34 / 37.70 / 0.01 / 23.02  
      void paddle::operators::KeBNBackwardScaleBias...  376     - / - / - / - / -                         126.37 / 0.34 / 1.84 / 0.03 / 22.73  
      void paddle::operators::KeBNBackwardData<floa...  376     - / - / - / - / -                         171.97 / 0.46 / 2.58 / 0.04 / 30.93  
    GpuMemcpyAsync:CPU->GPU                             64      0.71 / 0.01 / 0.02 / 0.01 / 1.88          0.08 / 0.00 / 0.00 / 0.00 / 0.01  
      MEMCPY_HtoD                                       64      - / - / - / - / -                         0.08 / 0.00 / 0.00 / 0.00 / 100.00  
    void phi::funcs::ConcatKernel_<float>(float con...  16      - / - / - / - / -                         6.40 / 0.40 / 0.53 / 0.34 / 0.95  
    void phi::funcs::ForRangeElemwiseOp<paddle::imp...  16      - / - / - / - / -                         6.23 / 0.39 / 0.56 / 0.27 / 0.93  
    ncclAllReduceRingLLKernel_sum_f32(ncclColl)         16      - / - / - / - / -                         95.02 / 5.94 / 7.56 / 4.75 / 14.17  
    void phi::funcs::SplitKernel_<float>(float cons...  16      - / - / - / - / -                         6.93 / 0.43 / 0.76 / 0.34 / 1.03  
  ```

  Operator Summary用于展示框架中算子(op)的执行信息。对于每一个Op，可以通过打印表单时候的op_detail选项控制是否打印出Op执行过程里面的子过程。同时展示每个子过程中的GPU上的活动，且子过程的活动算时间占比时以上层的时间为总时间。

- Kernel Summary
  ```text
  (由于原始表单较长，这里截取一部分进行展示)
  ---------------------------------------------------------------Kernel Summary---------------------------------------------------------------
  Time unit: ms
  ------------------------------------------------------------------------------------------  ------  ----------------------------------------  
  Name                                                                                        Calls   GPU Total / Avg / Max / Min / Ratio(%)  
  ------------------------------------------------------------------------------------------  ------  ----------------------------------------  
  void paddle::operators::KeNormAffine<float, (paddle::experimental::DataLayout)2>            376     362.11 / 0.96 / 5.43 / 0.09 / 12.97  
  ncclAllReduceRingLLKernel_sum_f32(ncclColl)                                                 784     257.23 / 0.33 / 37.70 / 0.01 / 9.22  
  maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile418n_nt                                       72      176.84 / 2.46 / 3.35 / 1.90 / 6.34  
  void paddle::operators::KeBNBackwardData<float, (paddle::experimental::DataLayout)2>        376     171.97 / 0.46 / 2.58 / 0.04 / 6.16  
  maxwell_scudnn_128x128_stridedB_splitK_interior_nn                                          184     167.03 / 0.91 / 2.28 / 0.19 / 5.99  
  void paddle::operators::KeBackwardLocalStats<float, 256, (paddle::experimental::DataLay...  376     129.62 / 0.34 / 1.83 / 0.04 / 4.64  
  void paddle::operators::KeBNBackwardScaleBias<float, 256, (paddle::experimental::DataLa...  376     126.37 / 0.34 / 1.84 / 0.03 / 4.53  
  void phi::funcs::VectorizedElementwiseKernel<float, phi::funcs::CudaReluGradFunctor<flo...  216     115.61 / 0.54 / 2.31 / 0.07 / 4.14  
  void paddle::operators::math::KernelDepthwiseConvFilterGradSp<float, 1, 1, 3, (paddle::...  72      113.87 / 1.58 / 2.04 / 1.36 / 4.08  
  maxwell_scudnn_128x128_stridedB_interior_nn                                                 200     105.10 / 0.53 / 0.97 / 0.09 / 3.77  
  maxwell_scudnn_128x128_relu_interior_nn                                                     184     103.17 / 0.56 / 0.98 / 0.12 / 3.70  
  maxwell_scudnn_winograd_128x128_ldg1_ldg4_tile228n_nt                                       48      90.87 / 1.89 / 2.09 / 1.69 / 3.26  
  maxwell_scudnn_128x128_stridedB_small_nn                                                    24      87.58 / 3.65 / 4.00 / 3.53 / 3.14  
  ```
  Kernel Summary用于展示在GPU执行的kernel的信息。

- Memory Manipulation Summary
  ```text
  -------------------------------------------------Memory Manipulation Summary-------------------------------------------------
  Time unit: ms
  ---------------------------------  ------  ----------------------------------------  ----------------------------------------  
  Name                               Calls   CPU Total / Avg / Max / Min / Ratio(%)    GPU Total / Avg / Max / Min / Ratio(%)  
  ---------------------------------  ------  ----------------------------------------  ----------------------------------------  
  GpuMemcpySync:GPU->CPU             48      1519.87 / 31.66 / 213.82 / 0.02 / 30.73   0.07 / 0.00 / 0.00 / 0.00 / 0.00  
  GpuMemcpyAsync:CPU->GPU            216     2.85 / 0.01 / 0.04 / 0.01 / 0.06          0.29 / 0.00 / 0.00 / 0.00 / 0.01  
  GpuMemcpyAsync(same_gpu):GPU->GPU  168     3.61 / 0.02 / 0.05 / 0.01 / 0.07          0.33 / 0.00 / 0.01 / 0.00 / 0.01  
  GpuMemcpySync:CUDAPinned->GPU      40      713.89 / 17.85 / 85.79 / 0.04 / 14.44     29.11 / 0.73 / 3.02 / 0.00 / 1.03  
  BufferedReader:MemoryCopy          6       40.17 / 6.69 / 7.62 / 5.87 / 0.81         0.00 / 0.00 / 0.00 / 0.00 / 0.00  
  ---------------------------------  ------  ----------------------------------------  ----------------------------------------
  ```

  Memory Manipulation Summary用于展示框架中调用内存操作所花费的时间。


- UserDefined Summary
  ```text
  ------------------------------------------UserDefined Summary------------------------------------------
  Time unit: ms
  -----------  ------  ----------------------------------------  ----------------------------------------  
  Name         Calls   CPU Total / Avg / Max / Min / Ratio(%)    GPU Total / Avg / Max / Min / Ratio(%)  
  -----------  ------  ----------------------------------------  ----------------------------------------  
  --------------------------------------Thread: All threads merged---------------------------------------
  MyRecord     8       0.15 / 0.02 / 0.02 / 0.02 / 0.00          0.00 / 0.00 / 0.00 / 0.00 / 0.00  
  -----------  ------  ----------------------------------------  ----------------------------------------  
  ```


  UserDefined Summary用于展示用户自定义记录的Event所花费的时间。

### Benchmark信息
benckmark信息用于展示模型的吞吐量以及时间开销。
```text
============================================Perf Summary============================================
Reader Ratio: 0.989%
Time Unit: s, IPS Unit: steps/s
|                 |       avg       |       max       |       min       |
|   reader_cost   |     0.00010     |     0.00011     |     0.00009     |
|    batch_cost   |     0.00986     |     0.00798     |     0.00786     |
|       ips       |    101.41524    |    127.25977    |    125.29320    |
```
其中ReaderRatio表示数据读取部分占batch迭代过程的时间占比，reader_cost代表数据读取时间，batch_cost代表batch迭代的时间，ips表示每秒能迭代多少次，即跑多少个batch。


## 更多细节

关于paddle.profiler模块更详细的使用说明，可以参考[API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/profiler/Overview_cn.html)。目前Paddle的性能分析工具主要还只提供时间方面的分析，之后会提供更多信息的收集来辅助做更全面的分析，如提供显存分析来监控显存泄漏问题。此外，Paddle的可视化工具VisualDL正在对Profiler的数据展示进行开发，敬请期待。
