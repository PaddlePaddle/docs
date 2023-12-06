## [ torch 参数更多 ] torch.distributed.barrier
### [torch.distributed.barrier](https://pytorch.org/docs/stable/distributed.html?highlight=barrier#torch.distributed.barrier)

```python
torch.distributed.barrier(group=None, async_op=False, device_ids=None)
```

### [paddle.distributed.barrier](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/barrier_cn.html)

```python
paddle.distributed.barrier(group=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                  |
| ------------- | ------------ | ------------------------------------------------------|
| group         | group        | 进程组编号。                                           |
| async_op      | -            | 是否是异步算子，Paddle 无此参数，暂无转写方式。             |
| device_ids    | -            | 设备 id，Paddle 无此参数，暂无转写方式。                  |
