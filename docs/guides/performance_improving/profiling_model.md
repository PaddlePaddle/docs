# 模型性能分析
Paddle profiler模块是paddle框架自带的低开销性能分析器，用于辅助用户对模型运行过程中的性能数据进行分析。用户可以通过性能分析器在程序运行过程中收集到的各种性能数据所导出的timeline和相关统计指标，来对程序的执行瓶颈进行判断分析，并寻求优化方案来获得性能的提升。用户可以识别到的问题一般有GPU“饥饿”所导致的利用率低，不必要的GPU同步，不充分的GPU并行，或者是算法计算复杂度太高等。

在这篇文档中，我们将对如何使用paddle profiler做性能分析进行说明，介绍程序所输出的timeline和统计表单，以及使用Profiler输出benchmark相关信息，最后通过一个简单的使用案例来阐述如何利用性能分析工具进行性能调试。

## 内容
- [Paddle&nbsp;Profiler使用介绍](#paddle-profiler)
- [Timeline展示](#timeline)
- [统计表单展示](#tongjibiaodanzhanshi)
- [Benchmark信息](#benchmark)
- [使用案例](#shiyonganli)

### Paddle&nbsp;Profiler使用介绍
关于paddle.profiler模块的API说明，在API文档的[paddle.profiler](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/profiler/Overview_cn.html)中, 这里主要根据常用使用场景来进行示例说明。

1、 将paddle.profiler.Profiler作为Context Manager, 对所包含的代码块进行性能分析
  - 对某一段batch的训练过程进行性能分析，如batch [2,10），前闭后开区间
    ```python
    import paddle
    import paddle.profiler as profiler # 引入paddle.profiler包

    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=1e-4, parameters = linear.parameters())
    # 初始化Profiler对象，并使用with语句
    with profiler.Profiler(
             targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
             scheduler=(2, 10),
             on_trace_ready=profiler.export_chrome_tracing('./profiler_demo')) as prof:
      # 进入with代码块，对如下代码块进行性能分析
      for i in range(30):
          data = paddle.randn(shape=[26])
          data = paddle.reshape(data, [2, 13])
          out = linear(data)
          out.backward()
          momentum.step()
          momentum.clear_grad()
          prof.step() # 每迭代一个step(batch), prof也调用step(), 告知Profiler进入了下一个step(batch)
    # 离开with代码块，性能分析结束
    prof.summary() # 打印统计表单
    ```
    该段代码会对训练过程的batch [2, 10), 共8个batch在CPU和GPU上的性能数据进行采集，并将收集到的性能数据以chrome tracing timeline的格式保存在profiler_demo文件夹中，最后对收集到的性能数据进行统计分析打印到终端。

  - 对多段batch的训练过程进行性能分析，如[2, 5], [8, 11], [14, 17]
    ```python
    import paddle
    import paddle.profiler as profiler # 引入paddle.profiler包
    
    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=1e-4, parameters = linear.parameters())
    # 初始化Profiler对象，并使用with语句
    with profiler.Profiler(
             targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
             scheduler=profiler.make_scheduelr(closed=1, ready=1, record=4, repeat=3),
             on_trace_ready=profiler.export_chrome_tracing('./profiler_demo')) as prof:
      # 进入with代码块，对如下代码块进行性能分析
      for i in range(30):
          data = paddle.randn(shape=[26])
          data = paddle.reshape(data, [2, 13])
          out = linear(data)
          out.backward()
          momentum.step()
          momentum.clear_grad()
          prof.step() # 每迭代一个step(batch), prof也调用step(), 告知Profiler进入了下一个step(batch)
    # 离开with代码块，性能分析结束
    ```
    该段代码会对训练过程的batch [2, 5] [8, 11] [14 17], 共3段batch在CPU和GPU上的性能数据进行分开采集，并且将收集到的性能数据以chrome tracing timeline的格式保存在profiler_demo文件夹中，一共会产生3个文件，每一个文件中分别存储某一段batch中所采集到的性能数据。

  - 对所有batch的训练过程进行性能分析
    ```python
    import paddle
    import paddle.profiler as profiler # 引入paddle.profiler包
    
    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=1e-4, parameters = linear.parameters())
    # 初始化Profiler对象，并使用with语句
    with profiler.Profiler() as prof:
      # 进入with代码块，对如下代码块进行性能分析
      for i in range(20):
          data = paddle.randn(shape=[26])
          data = paddle.reshape(data, [2, 13])
          out = linear(data)
          out.backward()
          momentum.step()
          momentum.clear_grad()
          prof.step() # 每迭代一个step(batch), prof也调用step(), 告知Profiler进入了下一个step(batch)
    # 离开with代码块，性能分析结束
    prof.summary() # 打印统计表单
    ```
    该段代码会对整个训练过程性能数据进行采集（默认的scheduler参数会让Profiler始终保持收集数据的RECORD状态），即batch [0, 20)在CPU和GPU上的性能数据（默认的targets参数会判断是否支持GPU数据的采集，支持则自动开启）, 并将收集到的性能数据以chrome tracing timeline的格式保存在profiler_log文件夹(默认的on_trace_ready参数会将日志文件保存到profiler_log文件夹)中，最后对收集到的性能数据进行统计分析打印到终端。在正常使用中不推荐这种方式，因为采集性能数据的batch太多有可能会耗尽所有的内存，并且导出的文件也会非常大，一般采几个batch的性能数据就能够对整个程序的运行情况有个判断了，没必要采集所有数据。


2、 手动调用paddle.profiler.Profiler的start, step, stop方法来对代码进行性能分析

  - 对某一段batch的训练过程进行性能分析，如第[2,10）个batch，前闭后开区间
    ```python
    import paddle
    import paddle.profiler as profiler # 引入paddle.profiler包

    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=1e-4, parameters = linear.parameters())

    # 初始化Profiler对象
    prof = profiler.Profiler(
               targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
               scheduler=(2, 10),
               on_trace_ready=profiler.export_chrome_tracing('./profiler_demo'))
    prof.start() # 调用start()方法，告知Profiler进入第0个step(batch), 进入性能分析过程
    for i in range(30):
        data = paddle.randn(shape=[26])
        data = paddle.reshape(data, [2, 13])
        out = linear(data)
        out.backward()
        momentum.step()
        momentum.clear_grad()
        prof.step() # 每迭代一个step(batch), prof也调用step(), 告知Profiler进入了下一个step(batch)
    prof.stop() # 调用stop()方法，告知Profiler性能分析过程结束，Profiler进入CLOSED状态
    prof.summary() # 打印统计表单
    ```
    该段代码手动调用Profiler的start()和stop()来开启和关闭Profiler，其实在上述的with语句用法中，在进入with代码块和离开with代码块的时候，也是分别调用了这两个方法而已。
    用这种手动调用start()和stop()来代替使用with语句的方式，可以避免对所分析的代码块进行缩进。

3、 自定义scheduler来控制性能分析过程的跨度

  - 上述例子中，我们是通过将一个二元组tuple，如(2,10) 或者是通过make_scheduler接口来生成scheduler。实际上也可以自己来定义scheduler，比如定义一个收集所有batch的性能数据的scheduler

    ```python
    import paddle
    import paddle.profiler as profiler

    # 定义一个收集所有batch的scheduler，即当前不管是第几个step(batch), 都返回RECORD状态
    def my_scheduler(step):
      return profiler.ProfilerState.RECORD
    
    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=1e-4, parameters = linear.parameters())

    # 初始化Profiler对象
    prof = profiler.Profiler(
        targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
        scheduler=my_scheduler, # 放入自定义的scheduler
        on_trace_ready=profiler.export_chrome_tracing('./profiler_demo'))
    prof.start() # 调用start()方法，告知Profiler进入第0个step(batch), 进入性能分析过程
    for i in range(30):
        data = paddle.randn(shape=[26])
        data = paddle.reshape(data, [2, 13])
        out = linear(data)
        out.backward()
        momentum.step()
        momentum.clear_grad()
        prof.step() # 每迭代一个step(batch), prof也调用step(), 告知Profiler进入了下一个step(batch)
    prof.stop() # 调用stop()方法，告知Profiler性能分析过程结束，Profiler进入CLOSED状态
    prof.summary() # 打印统计表单
    ```

4、 自定义on_trace_ready来控制每一段性能分析过程结束后的动作

  - 当对多段batch的训练过程进行性能分析，如上述例子中的batch [2, 5], [8, 11], [14, 17]，如果在离开with语句块后加上prof.summary()进行打印，将只能打印最后一段batch, 即batch [14, 17]这段时间内所收集的性能数据的统计结果。这是因为Profiler只会持有最新返回的性能数据，如果当某一段batch的性能数据返回时，没有进行处理，那等
  下一段性能数据返回时，就会覆盖上一段的数据。这也是on_trace_ready参数的用处所在，既是提供给用户一种自定义后处理的方式，同时也是为了能够及时对每段返回的性能数据进行处理，
  当性能数据返回时，Profiler将会调用on_trace_ready回调函数进行处理。Profiler的默认on_trace_ready参数是profiler.export_chrome_tracing('./profiler_log/')，上述示例所填的on_trace_ready参数也基本是这一回调函数，所做的即是每当性能数据返回时，以chrome tracing timeline的格式导出到指定文件夹。
    对于这种多段batch的性能分析，如果需要对每一段的数据都导出到chrome tracing timeline，并且打印统计表单，我们可以自定义on_trace_ready回调函数:
    ```python
    import paddle
    import paddle.profiler as profiler

    # 定义一个回调函数处理Profiler, 每当一段batch的性能数据采集结束并返回，都会调用该回调函数进行处理
    def my_on_trace_ready(prof):
      callback = profiler.export_chrome_tracing('./profiler_demo')
      callback(prof) # 导出数据到chrome tracing timeline
      prof.summary() # 调用summary方法打印表单
    
    linear = paddle.nn.Linear(13, 5)
    momentum = paddle.optimizer.Momentum(learning_rate=1e-4, parameters = linear.parameters())

    # 初始化Profiler对象
    prof = profiler.Profiler(
        targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
        scheduler=(2, 10), 
        on_trace_ready=my_on_trace_ready) # 放入自定义的on_trace_ready回调函数
    prof.start() # 调用start()方法，告知Profiler进入第0个step(batch), 进入性能分析过程
    for i in range(30):
        data = paddle.randn(shape=[26])
        data = paddle.reshape(data, [2, 13])
        out = linear(data)
        out.backward()
        momentum.step()
        momentum.clear_grad()
        prof.step() # 每迭代一个step(batch), prof也调用step(), 告知Profiler进入了下一个step(batch)
    prof.stop() # 调用stop()方法，告知Profiler性能分析过程结束，Profiler进入CLOSED状态
    ```

5、 在Python脚本中自定义记录某一个代码片段的性能数据
  - 为了分析某一段代码所花费的时间，可以使用profiler.RecordEvent接口来进行打点记录
    ```python
      import paddle
      import paddle.profiler as profiler

      linear = paddle.nn.Linear(13, 5)
      momentum = paddle.optimizer.Momentum(learning_rate=1e-4, parameters = linear.parameters())
      # 初始化Profiler对象
      prof = profiler.Profiler(
          targets=[ProfilerTarget.CPU, ProfilerTarget.GPU], 
          scheduler=(2, 10), 
          on_trace_ready=my_on_trace_ready) # 放入自定义的on_trace_ready回调函数
      prof.start() # 调用start()方法，告知Profiler进入第0个step(batch), 进入性能分析过程
      for i in range(30):
          with profiler.RecordEvent("DataPrepare"):
            data = paddle.randn(shape=[26])
            data = paddle.reshape(data, [2, 13])
          out = linear(data)
          out.backward()
          momentum.step()
          momentum.clear_grad()
          prof.step() # 每迭代一个step(batch), prof也调用step(), 告知Profiler进入了下一个step(batch)
      prof.stop() # 调用stop()方法，告知Profiler性能分析过程结束，Profiler进入CLOSED状态
      prof.summary() # 打印统计表单
    ```
    该代码片段将会记录paddle.randn和paddle.reshape这两句代码所花费的时间，所自定义的名字"DataPrepare"将会出现在chrome tracing timeline以及统计表单中，方便对此代码片段的性能进行分析。注意，在正常情况下，无需对模型训练或推理过程的dataloader, forward, backward和optimizer部分的代码进行自定义打点记录，我们的Profiler已经默认对这些代码进行了记录。

6、 仅使用Profiler做benchmark有关的数据统计
  ```python
  import paddle
  import paddle.profiler as profiler

  class RandomDataset(paddle.io.Dataset):
      def __init__(self, num_samples):
          self.num_samples = num_samples

      def __getitem__(self, idx):
          image = paddle.rand(shape=[100], dtype='float32')
          label = paddle.randint(0, 10, shape=[1], dtype='int64')
          return image, label

      def __len__(self):
          return self.num_samples

  class SimpleNet(paddle.nn.Layer):
      def __init__(self):
          super(SimpleNet, self).__init__()
          self.fc = paddle.nn.Linear(100, 10)

      def forward(self, image, label=None):
          return self.fc(image)

  dataset = RandomDataset(20 * 4)
  simple_net = SimpleNet()
  opt = paddle.optimizer.SGD(learning_rate=1e-3,
                              parameters=simple_net.parameters())
  BATCH_SIZE = 4
  loader = paddle.io.DataLoader(
      dataset,
      batch_size=BATCH_SIZE)
  p = profiler.Profiler(timer_only=True) # 仅做benchmark有关的统计
  p.start()
  for i, (image, label) in enumerate(loader()):
      out = simple_net(image)
      loss = paddle.nn.functional.cross_entropy(out, label)
      avg_loss = paddle.mean(loss)
      avg_loss.backward()
      opt.minimize(avg_loss)
      simple_net.clear_gradients()
      p.step(num_samples=BATCH_SIZE) 
      if i % 10 == 0:
          step_info = p.step_info(unit='images') 
          print("Iter {}: {}".format(i, step_info)) # 打印到第i个batch的信息
  p.stop() # 打印总的benchmark表单
  ```
  这段代码会只开启Profiler的benchmark统计功能，用于输出模型的吞吐量和执行时间信息，而不开启详细性能数据的采集。如果只需要获得ips(iterations per second)的数据，
  而不关心各部分的详细性能，可以如上所示设置timer_only=True。

### Timeline展示
对于采集的性能数据，通过上述示例代码的方法导出为chrome tracing timeline格式的文件后，可以进行可视化分析。当前，所采用的可视化工具为google chrome浏览器里的tracing插件，可以按照如下方式进行查看
  <p align="center">
  <img src="https://user-images.githubusercontent.com/22424850/161976125-27838228-d1c2-48ec-a96b-03d8f1bdad65.gif"   width='80%' hspace='10'/>
  <br />
  Timeline使用Demo
  </p>
目前Timeline提供以下特性：

1. 查看CPU和GPU在不同线程或stream下的事件发生的时间线。将同一线程下所记录的数据分为Python层和C++层，以便根据需要进行折叠和展开。对于有名字的线程，标注线程名字。
2. 所展示的事件名字上标注事件所持续的时间，点击具体的事件，可在下方的说明栏中看到更详细的事件信息。通过按键'w', 's'可进行放大和缩小，通过'a','d'可进行左移和右移。
3. 对于GPU上的事件，可以通过点击下方的'launch'链接查看所发起它的CPU上的事件。



### 统计表单展示
统计表单负责对采集到的数据(Event)从多个不同的角度进行解读，也可以理解为对timeline进行一些量化的指标计算。
目前提供的Device Summary、Overview Summary、Model Summary、Distributed Summary、Operator Summary、Kernel Summary、Memory Manipulation Summary和UserDefined Summary的统计，
每个统计表单从不同的角度根据需要取出对应类型的性能数据进行统计计算。每种表单的统计内容简要叙述如下：

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
其中ReaderRatio表示数据读取占一个batch迭代过程的时间占比，reader_cost代表数据读取时间，batch_cost代表一个batch的时间，ips表示每秒能迭代多少次，即跑多少个batch。

### 使用案例

我们以一个比较简单的示例，来看性能分析工具是如何在调试程序性能中发挥作用。下面是Paddle的应用实践教学中关于[使用神经网络对cifar10进行分类](https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/cv/convnet_image_classification.html)的示例代码，我们加上了性能分析的代码
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
    def my_on_trace_ready(prof):
      callback = profiler.export_chrome_tracing('./profiler_demo')
      callback(prof)
      prof.stop() # 可以打印benchmark信息
      prof.summary(sorted_by=profiler.SortedKeys.GPUTotal)
    p = profiler.Profiler(scheduler = [3,14], on_trace_ready=my_on_trace_ready)
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
    p.stop()
```
通过分析第3-9个batch的性能数据，导出timeline和统计表单。
<p align="center">
<img src="https://user-images.githubusercontent.com/22424850/165498308-734b4978-252e-45fc-8376-aaf8eb8a4270.png"   width='80%' hspace='10'/>
<br />
</p>

```text
-----------------------------------------------Model Summary-----------------------------------------------
Time unit: ms
---------------  ------  ----------------------------------------  ----------------------------------------  
Name             Calls   CPU Total / Avg / Max / Min / Ratio(%)    GPU Total / Avg / Max / Min / Ratio(%)    
---------------  ------  ----------------------------------------  ----------------------------------------  
ProfileStep      11      293.39 / 26.67 / 30.42 / 25.42 / 100.00   13.25 / 1.20 / 1.21 / 1.20 / 100.00       
  Dataloader     11      144.09 / 13.10 / 15.09 / 12.05 / 49.11    0.00 / 0.00 / 0.00 / 0.00 / 0.00          
  Forward        11      50.26 / 4.57 / 5.34 / 4.22 / 17.13        3.96 / 0.36 / 0.37 / 0.36 / 29.73         
  Backward       11      20.49 / 1.86 / 2.26 / 1.55 / 6.99         8.13 / 0.74 / 0.74 / 0.73 / 61.30         
  Optimization   11      34.52 / 3.14 / 3.32 / 2.52 / 11.77        0.67 / 0.06 / 0.06 / 0.06 / 5.03          
  Others         -       44.03 / - / - / - / 15.01                 0.52 / - / - / - / 3.94                   
---------------  ------  ----------------------------------------  ---------------------------------------- 
```
benchmark工具输出的信息如下所示（由于打点位置不同和数据处理上的差异，benchmark输出的Reader Ratio和上面统计表单中Dataloader的比例不完全一致）
```text
============================================Perf Summary============================================
Reader Ratio: 38.304%
Time Unit: s, IPS Unit: steps/s
|                 |       avg       |       max       |       min       |
|   reader_cost   |     0.01236     |     0.01277     |       inf       |
|    batch_cost   |     0.03228     |     0.02624     |     0.02544     |
|       ips       |     30.98171    |     39.30185    |     38.11149    |
```

从timeline和统计表单中可以看到，dataloader占了执行过程的很大比重，甚至接近了50%。通过分析程序发现，这是由于模型本身比较简单，需要的计算量小，再加上dataloader
准备数据时只用了单线程来读取，使得程序近乎没有并行操作，导致dataloader占比过大。通过对程序做如下修改，将dataloader的num_workers设置为4，使得能有多个线程并行读取数据。
```python
train_loader = paddle.io.DataLoader(cifar10_train,
                                    shuffle=True,
                                    batch_size=batch_size,
                                    num_workers=4)
```
重新对程序进行性能分析，新的timeline和统计表单如下所示
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
ProfileStep      11      93.45 / 8.50 / 12.00 / 7.78 / 100.00      13.26 / 1.21 / 1.22 / 1.19 / 100.00       
  Dataloader     11      1.70 / 0.15 / 0.55 / 0.11 / 1.82          0.00 / 0.00 / 0.00 / 0.00 / 0.00          
  Forward        11      32.25 / 2.93 / 5.56 / 2.52 / 34.51        3.84 / 0.35 / 0.35 / 0.35 / 30.73         
  Backward       11      15.43 / 1.40 / 2.09 / 1.32 / 16.51        8.27 / 0.75 / 0.76 / 0.74 / 60.58         
  Optimization   11      17.55 / 1.60 / 1.95 / 1.55 / 18.78        0.66 / 0.06 / 0.06 / 0.06 / 4.84          
  Others         -       26.52 / - / - / - / 28.38                 0.53 / - / - / - / 3.86                   
---------------  ------  ----------------------------------------  ----------------------------------------   
```
benchmark工具输出的信息如下所示
```text
============================================Perf Summary============================================
Reader Ratio: 0.989%
Time Unit: s, IPS Unit: steps/s
|                 |       avg       |       max       |       min       |
|   reader_cost   |     0.00010     |     0.00011     |     0.00009     |
|    batch_cost   |     0.00986     |     0.00798     |     0.00786     |
|       ips       |    101.41524    |    127.25977    |    125.29320    |
```
可以看到，从dataloader中取数据的时间大大减少，变成了平均只占一个step的1.8%，并且一个step所需要的时间也相应减少了。从benchmark工具给出的信息来看，ips也从平均30增长到了101，程序性能得到了极大的提升。
通过Profiler工具，您也可以使用RecordEvent对您所想要分析的程序片段进行监控，以此来寻找瓶颈点进行优化。

**Note**: 目前Paddle的性能分析工具主要还只提供时间方面的分析，之后会提供更多信息的收集来辅助做更全面的分析，如提供显存分析来监控显存泄漏问题。此外，Paddle的可视化工具VisualDL正在对Profiler的数据展示进行开发，敬请期待。

