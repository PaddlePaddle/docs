## [torch 参数更多]torch.distributed.irecv

### [torch.distributed.irecv](https://pytorch.org/docs/stable/distributed.html?highlight=send#torch.distributed.irecv)

```python
torch.distributed.irecv(tensor, src=None, group=None, tag=0)
```

### [paddle.distributed.irecv](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/irecv_cn.html)

```python
paddle.distributed.irecv(tensor, src=0, group=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle    | 备注                                                              |
| ------- | --------------- | ----------------------------------------------------------------- |
| tensor  | tensor          | 表示用于接收数据的 Tensor。                                               |
| src     | src             | 表示目标进程的 rank。                                                  |
| group   | group           | 表示执行该操作的进程组实例。   |
| tag     | -               | 表示匹配接收标签，Paddle 无此参数，暂无转写方式。   |
