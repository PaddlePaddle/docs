## [torch 参数更多]torch.distributed.new_group

### [torch.distributed.new_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.new_group)

```python
torch.distributed.new_group(ranks=None, timeout=datetime.timedelta(seconds=1800), backend=None, pg_options=None)
```

### [paddle.distributed.new_group](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/new_group_cn.html)

```python
paddle.distributed.new_group(ranks=None, backend=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                      |
| ---------- | ------------ | ----------------------------------------- |
| ranks      | ranks        | 用于新建通信组的全局 rank 列表。          |
| timeout    | -            | 进程组执行超时时间，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| backend    | backend      | 用于新建通信组的后端支持。                |
| pg_options | -            | 进程组选项，Paddle 无此参数，暂无转写方式。         |
