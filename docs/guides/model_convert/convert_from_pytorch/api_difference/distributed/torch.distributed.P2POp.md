## [torch 参数更多]torch.distributed.P2POp

### [torch.distributed.P2POp](https://pytorch.org/docs/stable/distributed.html#torch.distributed.P2POp)

```python
torch.distributed.P2POp(op, tensor, peer, group=None, tag=0)
```

### [paddle.distributed.P2POp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/Overview_cn.html#paddle-distributed)

```python
paddle.distributed.P2POp(op, tensor, peer, group=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle    | 备注                                                              |
| ------- | --------------- | ----------------------------------------------------------------- |
| op  | op          | 表示操作类型。                                               |
| tensor  | tensor          | 表示发送或接收的 Tensor。                                               |
| peer     | peer             | 表示目标进程的 rank。                                                  |
| group   | group           | 指定通信的进程组。   |
| tag     | -               | 表示匹配接收标签，Paddle 无此参数，暂无转写方式。   |
