## [torch 参数更多]torch.multiprocessing.spawn

### [torch.multiprocessing.spawn](https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn)

```python
torch.multiprocessing.spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')
```

### [paddle.distributed.spawn](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/spawn_cn.html#spawn)

```python
paddle.distributed.spawn(func, args=(), nprocs=- 1, join=True, daemon=False, **options)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                                |
| ------------ | ------------ | ------------------------------------------------------------------- |
| fn         | func         | Python 函数。                                                       |
| args       | args           | 函数 func 的输入参数。                                              |
| nprocs            | nprocs            | 启动进程的数目。 与 Pytorch 默认值不同， Paddle 应设置为 `1`。                                          |
| join | join            | 对所有启动的进程执行阻塞的 join，等待进程执行结束。                                   |
| start_method       | -            | 启动方式。 Pytorch 已弃用， Paddle 无此参数。可直接删除。 |
| -       | options           | 其他初始化并行执行环境的配置选项。 Pytorch 无此参数， Paddle 保持默认即可。 |
