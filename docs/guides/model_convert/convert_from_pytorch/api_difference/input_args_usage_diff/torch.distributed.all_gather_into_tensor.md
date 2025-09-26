## [ 输入参数用法不一致 ]torch.distributed.all_gather_into_tensor

### [torch.distributed.all_gather_into_tensor](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor)

```python
torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)

```

### [paddle.distributed.all_gather](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/all_gather_cn.html#all-gather)

```python
paddle.distributed.all_gather(tensor_list, tensor, group=None, sync_op=True)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                          |
| -------- | ------------ | --------------------------------------------- |
| output_tensor |      tensor_list       | 表示用于保存聚合结果的张量，torch 为 Tensor， Paddle 为 list，需要转写。 |
| input_tensor      | tensor          | 表示待聚合的张量，仅参数名不一致。                  |
| group    | group        | 表示执行该操作的进程组实例。                            |
| async_op    | sync_op      | torch 为是否异步操作，Paddle 为是否同步操作，转写方式取反即可。 |

### 转写示例
#### output_tensor：输出张量
```python
# PyTorch 写法
import torch.distributed as dist
dist.all_gather_into_tensor(output_tensor=output_tensor, input_tensor=input_tensor)

# Paddle 写法
import paddle.distributed as dist
tensor_list = []
dist.all_gather(tensor_list=tensor_list, tensor=input_tensor)
if paddle.distributed.get_world_size() * input_tensor.shape[0] == output_tensor.shape[0]:
    output_tensor = paddle.concat(tensor_list, axis=0)
else:
    output_tensor = paddle.stack(tensor_list, axis=0)
```
