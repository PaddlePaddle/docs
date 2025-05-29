## [ 输入参数用法不一致 ]torch.distributed.all_to_all_single

### [torch.distributed.all_to_all_single](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single)

```python
torch.distributed.all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False)
```

### [paddle.distributed.alltoall_single](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/alltoall_single_cn.html#alltoall-single)

```python
paddle.distributed.alltoall_single(out_tensor, in_tensor, in_split_sizes=None, out_split_sizes=None, group=None, sync_op=True)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch            | PaddlePaddle    | 备注                                                            |
| ------------------ | --------------- | --------------------------------------------------------------- |
| output | out_tensor | 用于保存操作结果的 Tensor，仅参数名不一致。               |
| input  | in_tensor  | 输入的 Tensor， 仅参数名不一致。               |
| output_split_sizes              | out_split_sizes           | 对输出 Tensor 的 dim[0] 进行切分的大小，仅参数名不一致。 |
| input_split_sizes           | in_split_sizes         | 对输入 Tensor 的 dim[0] 进行切分的大小，仅参数名不一致。  |
| group              | group           | new_group 返回的 Group 实例，或者设置为 None 表示默认地全局组。 |
| async_op           | sync_op         | torch 为是否异步操作，Paddle 为是否同步操作，转写方式取反即可。 |


### 转写示例
#### async_op：是否为异步操作

```python
# PyTorch 写法:
torch.distributed.all_to_all_single(output=output, input=input, async_op=True)

# Paddle 写法:
paddle.distributed.alltoall_single(out_tensor=output, in_tensor=input, sync_op=False)
```
