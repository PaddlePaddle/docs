## torch.distributed.barrier
### [torch.distributed.barrier](https://pytorch.org/docs/stable/distributed.html?highlight=barrier#torch.distributed.barrier)

```python
torch.distributed.barrier(group=None, async_op=False, device_ids=None)
```

### [paddle.distributed.barrier](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/barrier_cn.html)

```python
paddle.distributed.barrier(group=0)
```

两者功能一致，torch 参数更多，具体差异如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| group         | group        | 进程组编号                                 |
| async_op      | -            | 是否是异步算子，paddle 无此参数                                   |
| device_ids    | -            | 设备 id，paddle 无此参数                                     |
