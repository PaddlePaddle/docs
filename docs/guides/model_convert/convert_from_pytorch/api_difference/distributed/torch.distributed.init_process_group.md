## [ torch 参数更多 ] torch.distributed.init_process_group
### [torch.distributed.init_process_group](https://pytorch.org/docs/stable/distributed.html?highlight=init_process#torch.distributed.init_process_group)

```python
torch.distributed.init_process_group(backend='nccl', init_method=None, timeout=datetime.timedelta(seconds=1800), world_size=-1, rank=-1, store=None, group_name='', pg_options=None)
```

### [paddle.distributed.init_parallel_env](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/init_parallel_env_cn.html)

```python
paddle.distributed.init_parallel_env()
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| backend       | -        | backend 配置，paddle 无此参数，暂无转写方式。                   |
| init_method   | -        | 初始化方法，paddle 无此参数，暂无转写方式。                      |
| timeout       | -        | 超时配置，paddle 无此参数，暂无转写方式。                        |
| world_size    | -        | 进程数量，paddle 无此参数，暂无转写方式。                        |
| rank          | -        | 当前进程所在的 gpu，paddle 无此参数，暂无转写方式。               |
| store         | -        | 信息交换的配置，paddle 无此参数，暂无转写方式。                   |
| group_name    | -        | 组名，paddle 无此参数，暂无转写方式。                           |
| pg_options    | -        | 进程组配置，paddle 无此参数，暂无转写方式。                      |
