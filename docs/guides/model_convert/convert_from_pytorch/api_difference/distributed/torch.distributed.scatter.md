## [torch 参数更多]torch.distributed.scatter

### [torch.distributed.scatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.scatter)

```python
torch.distributed.scatter(tensor, scatter_list=None, src=0, group=None, async_op=False)
```

### [paddle.distributed.scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/scatter_cn.html)

```python
paddle.distributed.scatter(tensor, tensor_list=None, src=0, group=0)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                          |
| ------------ | ------------ | --------------------------------------------- |
| tensor       | tensor       | 操作的输出 Tensor。                           |
| scatter_list | tensor_list  | 操作的输入 Tensor 列表，仅参数名不一致。      |
| src          | src          | 操作的源进程号。                              |
| group        | group        | 工作的进程组编号。                            |
| async_op     | -            | 是否异步操作，Paddle 无此参数，暂无转写方式。 |
