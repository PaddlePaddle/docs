## [ 输入参数用法不一致 ]torch.distributed.all_gather_into_tensor

### [torch.distributed.all_gather_into_tensor](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor)

```python
torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)

```

### [paddle.distributed.stream.all_gather](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/stream/all_gather_cn.html#all-gather)

```python
paddle.distributed.stream.all_gather(tensor_or_tensor_list, tensor, group=None, sync_op=True, use_calc_stream=False)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                          |
| -------- | ------------ | --------------------------------------------- |
| output_tensor |      tensor_or_tensor_list       | 表示用于保存聚合结果的张量，仅参数名不一致。 |
| input_tensor      | tensor          | 表示待聚合的张量，仅参数名不一致。                  |
| group    | group        | 表示执行该操作的进程组实例。                            |
| async_op    | sync_op      | torch 为是否异步操作，Paddle 为是否同步操作，转写方式取反即可。 |
| -    | use_calc_stream      | 该操作是否在计算流上进行，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例
#### async_op：是否为异步操作
```python
# PyTorch 写法
import torch.distributed as dist
dist.all_gather_into_tensor(output_tensor=output_tensor, input_tensor=data, async_op=True)

# Paddle 写法
import paddle.distributed as dist
dist.stream.all_gather(tensor_or_tensor_list=output_tensor, tensor=data, sync_op=False)
```
