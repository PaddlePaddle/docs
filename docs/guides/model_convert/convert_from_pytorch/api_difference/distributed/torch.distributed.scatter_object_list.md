## [仅参数名不一致]torch.distributed.scatter_object_list

### [torch.distributed.scatter_object_list](https://pytorch.org/docs/stable/distributed.html#torch.distributed.scatter_object_list)

```python
torch.distributed.scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=0, group=None)
```

### [paddle.distributed.scatter_object_list](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/scatter_object_list_cn.html#scatter-object-list)

```python
paddle.distributed.scatter_object_list(out_object_list, in_object_list, src=0, group=None)
```

其中 PyTorch 和 Paddle 功能一致，仅参数名不一致，具体如下：

### 参数映射

| PyTorch                    | PaddlePaddle    | 备注                                         |
| -------------------------- | --------------- | -------------------------------------------- |
| scatter_object_output_list | out_object_list | 用于接收数据的 object 列表，仅参数名不一致。 |
| scatter_object_input_list  | in_object_list  | 将被分发的 object 列表，仅参数名不一致。     |
| src                        | src             | 目标进程的 rank。                            |
| group                      | group           | 执行该操作的进程组实例。                     |
