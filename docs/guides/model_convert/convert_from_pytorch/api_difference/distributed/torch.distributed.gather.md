## [torch 参数更多]torch.distributed.gather

### [torch.distributed.gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.gather)

```python
torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)
```

### [paddle.distributed.gather](https://github.com/PaddlePaddle/Paddle/blob/c8ccc9b154632ef41ade1b8e97b87d54fde7e8f8/python/paddle/distributed/communication/gather.py#L20C71-L20C71)

```python
paddle.distributed.gather(tensor, gather_list=None, dst=0, group=None, sync_op=True)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                            |
| ----------- | ------------ | --------------------------------------------------------------- |
| tensor      | tensor       | 操作的输入 Tensor。                                             |
| gather_list | gather_list  | 操作的输出 Tensor 列表。                                        |
| dst         | dst          | 表示目标进程的 rank。                                           |
| group       | group        | 工作的进程组编号。                                              |
| async_op    | sync_op      | torch 为是否异步操作，Paddle 为是否同步操作，转写方式取反即可。 |
