## [ torch 参数更多 ]torch.distributed.rpc.init_rpc

### [torch.distributed.rpc.init\_rpc](https://pytorch.org/docs/stable/rpc.html#torch.distributed.rpc.init_rpc)

```python
torch.distributed.rpc.init_rpc(name, backend=None, rank=-1, world_size=None, rpc_backend_options=None)
```

### [paddle.distributed.rpc.init\_rpc](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/rpc/init_rpc_cn.html#init-rpc)

```python
paddle.distributed.rpc.init_rpc(name, rank=None, world_size=None, master_endpoint=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch             | PaddlePaddle        | 备注 |
| ------------------- | ------------------- | -- |
| name                | name                | worker 名字。 |
| backend             | -                   | RPC 后台实现类型，Paddle 无此参数，暂无转写方式。 |
| rank                | rank                | worker 的 ID，Paddle 与 PyTorch 默认值不同，Paddle 应设置为 -1。 |
| world_size          | world_size          | workers 的数量。 |
| rpc_backend_options | -                   | 传递给 worker 创建时的配置项，Paddle 无此参数，暂无转写方式。 |
| -                   | master_endpoint     | master 的 IP 地址，PyTorch 无此参数，Paddle 保持默认即可。 |
