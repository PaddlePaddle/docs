## [torch 参数更多]torch.distributed.all_gather

### [torch.distributed.all_gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather)

```python
torch.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)
```

### [paddle.distributed.all_gather](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/all_gather_cn.html)

```python
paddle.distributed.all_gather(tensor_list, tensor, group=0)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                          |
| ----------- | ------------ | --------------------------------------------- |
| tensor_list | tensor_list  | 操作的输出 Tensor 列表。                      |
| tensor      | tensor       | 操作的输入 Tensor。                           |
| group       | group        | 工作的进程组编号。                            |
| async_op    | -            | 是否异步操作，Paddle 无此参数，暂无转写方式。 |
