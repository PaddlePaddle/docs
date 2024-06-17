## [ 输入参数用法不一致 ]torch.distributed.all_to_all

### [torch.distributed.all_to_all](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all)

```python
torch.distributed.all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False)
```

### [paddle.distributed.alltoall](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/alltoall_cn.html)

```python
paddle.distributed.alltoall(in_tensor_list, out_tensor_list, group=None, sync_op=True)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch            | PaddlePaddle    | 备注                                                            |
| ------------------ | --------------- | --------------------------------------------------------------- |
| output_tensor_list | out_tensor_list | 包含所有输出 Tensors 的一个列表，仅参数名不一致。               |
| input_tensor_list  | in_tensor_list  | 包含所有输入 Tensors 的一个列表，仅参数名不一致。               |
| group              | group           | new_group 返回的 Group 实例，或者设置为 None 表示默认地全局组。 |
| async_op           | sync_op         | torch 为是否异步操作，Paddle 为是否同步操作，转写方式取反即可。 |
