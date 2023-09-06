## [torch 参数更多]torch.cuda.comm.broadcast

### [torch.cuda.comm.broadcast](https://pytorch.org/docs/stable/generated/torch.cuda.comm.broadcast.html#torch.cuda.comm.broadcast)

```python
torch.cuda.comm.broadcast(tensor, devices=None, *, out=None)
```

### [paddle.distributed.broadcast](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/broadcast_cn.html)

```python
paddle.distributed.broadcast(tensor, src, group=None, sync_op=True)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                         |
| ------- | ------------ | ------------------------------------------------------------ |
| tensor  | tensor       | 在目标进程上为待广播的 tensor，在其他进程上为用于接收广播结果的 tensor。 |
| devices | src          | 发送源的进程编号。                                           |
| out     | -            | 表示输出的 Tensor ，Paddle 无此参数，需要转写。          |
| -       | group        | 工作的进程组编号，PyTorch 无此参数，Paddle 保持默认即可。    |
| -       | sync_op      | 该操作是否为同步操作。默认为 True，即同步操作。PyTorch 无此参数，Paddle 保持默认即可。    |

### 转写示例

#### out 参数：指定输出
``` python
# PyTorch 写法:
torch.cuda.comm.broadcast(x, 0, out=y)

# Paddle 写法:
paddle.assign(paddle.distributed.broadcast(x, 0) , y)
```
