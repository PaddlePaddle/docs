## [参数完全一致]torch.distributed.all_gather_object

### [torch.distributed.all_gather_object](https://pytorch.org/docs/2.0/distributed.html?highlight=all_gather_object#torch.distributed.all_gather_object)

```python
torch.distributed.all_gather_object(object_list, obj, group=None)
```

### [paddle.distributed.all_gather_object](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/all_gather_object_cn.html)

```python
paddle.distributed.all_gather_object(object_list, obj, group=None)
```

功能一致，参数完全一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                          |
| -------- | ------------ | --------------------------------------------- |
| object_list   | object_list       | 表示用于保存聚合结果的列表。                           |
| obj      | obj          | 表示待聚合的对象。                  |
| group    | group        | 表示执行该操作的进程组实例。                            |
