## [ torch 参数更多 ] torch.distributed.monitored_barrier
### [torch.distributed.monitored_barrier](https://pytorch.org/docs/stable/distributed.html#torch.distributed.monitored_barrier)

```python
torch.distributed.monitored_barrier(group=None, timeout=None, wait_all_ranks=False)
```

### [paddle.distributed.barrier](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/barrier_cn.html)

```python
paddle.distributed.barrier(group=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                  |
| ------------- | ------------ | ------------------------------------------------------|
| group         | group        | 进程组编号。                                           |
| timeout      | -            | 超时时间，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。             |
| wait_all_ranks    | -            | 是否等待所有进程超时后才报错，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。                  |
