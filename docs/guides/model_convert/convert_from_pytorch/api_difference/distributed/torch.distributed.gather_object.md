## [torch 参数更多]torch.distributed.gather_object

### [torch.distributed.gather_object](https://pytorch.org/docs/stable/distributed.html#torch.distributed.gather_object)

```python
torch.distributed.gather_object(obj, object_gather_list=None, dst=0, group=None)
```

### [paddle.distributed.all_gather_object](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/all_gather_object_cn.html#all-gather-object)

```python
paddle.distributed.all_gather_object(object_list, obj, group=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch            | PaddlePaddle | 备注                                       |
| ------------------ | ------------ | ------------------------------------------ |
| obj                | obj          | 待聚合的对象。                             |
| object_gather_list | object_list  | 用于保存聚合结果的列表，仅参数名不一致。   |
| dst                | -            | 目标 rank，Paddle 无此参数，暂无转写方式。 |
| group              | group        | 执行该操作的进程组实例。                   |
