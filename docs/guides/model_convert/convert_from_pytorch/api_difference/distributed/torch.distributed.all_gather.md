## [参数不一致]torch.distributed.all_gather

### [torch.distributed.all_gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather)

```python
torch.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)
```

### [paddle.distributed.all_gather](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/all_gather_cn.html)

```python
paddle.distributed.all_gather(tensor_list, tensor, group=None, sync_op=True)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                                                            |
| ----------- | ------------ | --------------------------------------------------------------- |
| tensor_list | tensor_list  | 操作的输出 Tensor 列表。                                        |
| tensor      | tensor       | 操作的输入 Tensor。                                             |
| group       | group        | 工作的进程组编号。                                              |
| async_op    | sync_op      | torch 为是否异步操作，Paddle 为是否同步操作，转写方式取反即可。 |
