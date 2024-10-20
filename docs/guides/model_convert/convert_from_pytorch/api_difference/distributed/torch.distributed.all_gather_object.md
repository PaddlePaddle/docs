## [参数完全一致]torch.distributed.all_gather_object

### [torch.distributed.all_gather_object](https://pytorch.org/docs/stable/distributed.html?highlight=all_gather_object#torch.distributed.all_gather_object)

```python
torch.distributed.all_gather_object(object_list, obj, group=None)
```

### [paddle.distributed.all_gather_object](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/distributed/all_gather_object_cn.html)

```python
paddle.distributed.all_gather_object(object_list, obj, group=None)
```

功能一致，参数几乎完全一致。但`object_list`的初始化方式不同。具体如下：

### 参数映射

| PyTorch  | PaddlePaddle | 备注                                          |
| -------- | ------------ | --------------------------------------------- |
| object_list |             | 表示用于保存聚合结果的列表。需初始化成与 `group` 等长的列表 |
|             | object_list | 表示用于保存聚合结果的列表。需初始化成空列表           |
| obj      | obj          | 表示待聚合的对象。                  |
| group    | group        | 表示执行该操作的进程组实例。                            |

### 转写示例

```python
# PyTorch 写法
import torch.distributed as dist
object_list = [{}, {}] # NOTE: world size is 2
if dist.get_rank() == 0:
    obj = {"foo": [1, 2, 3]}
else:
    obj = {"bar": [4, 5, 6]}
dist.all_gather_object(object_list, obj)

# Paddle 写法
import paddle.distributed as dist
object_list = [] # No need to pre-allocate
if dist.get_rank() == 0:
    obj = {"foo": [1, 2, 3]}
else:
    obj = {"bar": [4, 5, 6]}
dist.all_gather_object(object_list, obj)
```
