## [torch 参数更多]torch.distributed.recv

### [torch.distributed.recv](https://pytorch.org/docs/stable/distributed.html#torch.distributed.recv)

```python
torch.distributed.recv(tensor, src=None, group=None, tag=0)
```

### [paddle.distributed.recv](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/recv_cn.html)

```python
paddle.distributed.recv(tensor, src=0, group=None, sync_op=True)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle    | 备注                                                              |
| ------- | --------------- | ----------------------------------------------------------------- |
| tensor  | tensor          | 表示用于接收数据的 Tensor。                                               |
| src     | src             | 表示目标进程的 rank。                                                  |
| group   | group           | 表示执行该操作的进程组实例。   |
| tag     | -               | 表示匹配接收标签，Paddle 无此参数，暂无转写方式。   |
| -       | sync_op | 表示该操作是否为同步操作，PyTorch 无此参数，Paddle 保持默认即可。 |
