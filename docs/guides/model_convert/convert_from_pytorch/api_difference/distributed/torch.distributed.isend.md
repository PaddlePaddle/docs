## [torch 参数更多]torch.distributed.isend

### [torch.distributed.isend](https://pytorch.org/docs/2.0/distributed.html#torch.distributed.isend)

```python
torch.distributed.isend(tensor, dst, group=None, tag=0)
```

### [paddle.distributed.isend](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/isend_cn.html)

```python
paddle.distributed.isend(tensor, dst=0, group=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle    | 备注                                                              |
| ------- | --------------- | ----------------------------------------------------------------- |
| tensor  | tensor          | 表示待发送的 Tensor。                                               |
| dst     | dst             | 表示目标进程的 rank。                                                  |
| group   | group           | 表示执行该操作的进程组实例。   |
| tag     | -               | 表示匹配接收标签，Paddle 无此参数，暂无转写方式。                     |
