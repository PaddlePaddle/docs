## [torch 参数更多]torch.distributed.P2POp

### [torch.distributed.P2POp](https://pytorch.org/docs/stable/distributed.html#torch.distributed.P2POp)

```python
torch.distributed.P2POp(op, tensor, peer=None, group=None, tag=0)
```

### [paddle.distributed.P2POp](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/distributed/communication/batch_isend_irecv.py)

```python
paddle.distributed.P2POp(op, tensor, peer, group=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle    | 备注                                                              |
| ------- | --------------- | ----------------------------------------------------------------- |
| op  | op          | 表示执行的操作类型。                                               |
| tensor     | tensor             | 表示进行通信的张量。                                                  |
| peer   | peer           | 表示通信的目标进程的 rank。   |
| tag     | -               | 表示匹配接收标签，Paddle 无此参数，暂无转写方式。   |
