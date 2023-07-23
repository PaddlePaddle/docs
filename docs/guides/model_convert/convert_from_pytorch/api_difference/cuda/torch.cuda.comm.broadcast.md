## [torch 参数更多]torch.cuda.comm.broadcast

### [torch.cuda.comm.broadcast](https://pytorch.org/docs/stable/generated/torch.cuda.comm.broadcast.html#torch.cuda.comm.broadcast)

```python
torch.cuda.comm.broadcast(tensor, devices=None, *, out=None)
```

### [paddle.distributed.broadcast](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/broadcast_cn.html)

```python
paddle.distributed.broadcast(tensor, src, group=0)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                         |
| ------- | ------------ | ------------------------------------------------------------ |
| tensor  | tensor       | 如果当前进程编号是源，那么这个 Tensor 变量将被发送给其他进程。 |
| devices | src          | 发送源的进程编号。                                           |
| out     | -            | 表示输出的 Tensor ，Paddle 无此参数，需要进行转写。          |
| -       | group        | 工作的进程组编号，PyTorch 无此参数，Paddle 保持默认即可。    |

### 转写示例

#### out 参数：指定输出
``` python
# PyTorch 写法:
torch.cuda.comm.broadcast(x, 0, out=y)

# Paddle 写法:
paddle.assign(paddle.distributed.broadcast(x, 0) , y)
```
