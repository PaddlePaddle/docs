## [torch 参数更多]torch.distributed.rpc.shutdown

### [torch.distributed.rpc.shutdown](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.shutdown)

```python
torch.distributed.rpc.shutdown(graceful=True, timeout=0)
```

### [paddle.distributed.rpc.shutdown](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/rpc/shutdown_cn.html)

```python
paddle.distributed.rpc.shutdown()
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                          |
| -------- | ------------ | --------------------------------------------- |
| graceful | -            | 是否优雅关闭，Paddle 无此参数，暂无转写方式。 |
| timeout  | -            | 操作超时时间，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
