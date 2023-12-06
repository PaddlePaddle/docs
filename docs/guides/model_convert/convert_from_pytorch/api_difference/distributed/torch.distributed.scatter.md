## [参数不一致]torch.distributed.scatter

### [torch.distributed.scatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.scatter)

```python
torch.distributed.scatter(tensor, scatter_list=None, src=0, group=None, async_op=False)
```

### [paddle.distributed.scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/scatter_cn.html)

```python
paddle.distributed.scatter(tensor, tensor_list=None, src=0, group=None, sync_op=True)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                            |
| ------------ | ------------ | --------------------------------------------------------------- |
| tensor       | tensor       | 操作的输出 Tensor。                                             |
| scatter_list | tensor_list  | 操作的输入 Tensor 列表，仅参数名不一致。                        |
| src          | src          | 操作的源进程号。                                                |
| group        | group        | 工作的进程组编号。                                              |
| async_op     | sync_op      | torch 为是否异步操作，Paddle 为是否同步操作，转写方式取反即可。 |
