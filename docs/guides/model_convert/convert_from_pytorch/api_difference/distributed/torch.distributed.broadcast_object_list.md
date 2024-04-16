## [torch 参数更多]torch.distributed.broadcast_object_list

### [torch.distributed.broadcast_object_list](https://pytorch.org/docs/stable/distributed.html?highlight=broadcast_object_list#torch.distributed.broadcast_object_list)

```python
torch.distributed.broadcast_object_list(object_list, src=0, group=None, device=None)
```

### [paddle.distributed.broadcast_object_list](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/broadcast_object_list_cn.html)

```python
paddle.distributed.broadcast_object_list(object_list, src, group=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle    | 备注                                                              |
| ------- | --------------- | ----------------------------------------------------------------- |
| object_list  | object_list  | 表示在目标进程上为待广播的 object 列表，在其他进程上为用于接收广播结果的 object 列表。 |
| src     | src             | 表示目标进程的 rank。                                                  |
| group   | group           | 表示执行该操作的进程组实例。   |
| device     | -               | 表示如果不为空，则对象在被广播之前将被序列化并转换为 Tensor 后移动到设备上，Paddle 无此参数，暂无转写方式。   |
