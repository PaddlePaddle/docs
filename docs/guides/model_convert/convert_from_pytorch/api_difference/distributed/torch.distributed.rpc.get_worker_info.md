## [仅参数名不一致]torch.distributed.rpc.get_worker_info

### [torch.distributed.rpc.get_worker_info](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.get_worker_info)

```python
torch.distributed.rpc.get_worker_info(worker_name=None)
```

### [paddle.distributed.rpc.get_worker_info](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/rpc/get_worker_info_cn.html#cn-api-distributed-rpc-get-worker-info)

```python
paddle.distributed.rpc.get_worker_info(name)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射

| PyTorch     | PaddlePaddle | 备注                            |
| ----------- | ------------ | ------------------------------- |
| worker_name | name         | worker 的名字，仅参数名不一致。 |
