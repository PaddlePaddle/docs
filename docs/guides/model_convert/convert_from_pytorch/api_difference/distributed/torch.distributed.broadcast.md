## [ 输入参数用法不一致 ]torch.distributed.broadcast

### [torch.distributed.broadcast](https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast)

```python
torch.distributed.broadcast(tensor, src, group=None, async_op=False)
```

### [paddle.distributed.broadcast](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/broadcast_cn.html)

```python
paddle.distributed.broadcast(tensor, src, group=None, sync_op=True)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                                                                                   |
| -------- | ------------ | ------------------------------------------------------------------------------------------------------ |
| tensor   | tensor       | 如果当前进程编号是源，那么这个 Tensor 变量将被发送给其他进程，否则这个 Tensor 将接收源发送过来的数据。 |
| src      | src          | 发送源的进程编号。                                                                                     |
| group    | group        | 工作的进程组编号。                                                                                     |
| async_op | sync_op      | torch 为是否异步操作，Paddle 为是否同步操作，转写方式取反即可。                                        |
