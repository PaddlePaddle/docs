## [torch 参数更多]torch.nn.parallel.DistributedDataParallel

### [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel)

```python
torch.nn.parallel.DistributedDataParallel(module, device_ids=None, output_device=None, dim=0, broadcast_buffers=True, process_group=None, bucket_cap_mb=25, find_unused_parameters=False, check_reduction=False, gradient_as_bucket_view=False, static_graph=False)
```

### [paddle.DataParallel](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/DataParallel_cn.html)

```python
paddle.DataParallel(layers, strategy=None, comm_buffer_size=25, last_comm_buffer_size=1, find_unused_parameters=False)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch                 | PaddlePaddle           | 备注                                                                                     |
| ----------------------- | ---------------------- | ---------------------------------------------------------------------------------------- |
| module                  | layers                 | 需要通过数据并行方式执行的模型，仅参数名不一致。                                         |
| device_ids              | -                      | 设置输入设备，Paddle 无此参数，暂无转写方式。                                                      |
| output_device           | -                      | 设置输出设备，Paddle 无此参数，暂无转写方式。                                                      |
| dim                     | -                      | 进行运算的轴，Paddle 无此参数，暂无转写方式。                                                      |
| broadcast_buffers       | -                      | forward 开始时是否同步缓存，Paddle 无此参数，暂无转写方式。                                        |
| process_group           | -                      | 进程组，Paddle 无此参数，暂无转写方式。                                                            |
| bucket_cap_mb           | comm_buffer_size       | 它是通信调用（如 NCCLAllReduce）时，参数梯度聚合为一组的内存大小（MB），仅参数名不一致。 |
| find_unused_parameters  | find_unused_parameters | 是否在模型 forward 函数的返回值的所有张量中，遍历整个向后图。                            |
| check_reduction         | -                      | 已废弃，Paddle 无此参数，暂无转写方式。                                                            |
| gradient_as_bucket_view | -                      | 是否在 allreduce 通讯中配置梯度，Paddle 无此参数，暂无转写方式。                                   |
| static_graph            | -                      | 是否训练静态图，Paddle 无此参数，暂无转写方式。                                                    |
| -                       | strategy               | 已废弃，数据并行的策略，PyTorch 无此参数，Paddle 保持默认即可。                          |
| -                       | last_comm_buffer_size  | 它限制通信调用中最后一个缓冲区的内存大小，PyTorch 无此参数，Paddle 保持默认即可。        |
