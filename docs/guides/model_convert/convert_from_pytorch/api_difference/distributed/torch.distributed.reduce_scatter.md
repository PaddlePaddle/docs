## [ 输入参数用法不一致 ]torch.distributed.reduce_scatter

### [torch.distributed.reduce_scatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter)

```python
torch.distributed.reduce_scatter(output, input_list, op=<RedOpType.SUM: 0>, group=None, async_op=False)
```

### [paddle.distributed.reduce_scatter](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/reduce_scatter_cn.html#reduce-scatter)

```python
paddle.distributed.reduce_scatter(tensor, tensor_list, op=ReduceOp.SUM, group=None, sync_op=True)
```

其中 PyTorch 和 Paddle 功能一致，参数用法不一致，具体如下：

### 参数映射

| PyTorch    | PaddlePaddle | 备注                                                                   |
| ---------- | ------------ | ---------------------------------------------------------------------- |
| output     | tensor       | 用于接收数据的 tensor，仅参数名不一致。                                |
| input_list | tensor_list  | 将被规约和分发的 tensor 列表，仅参数名不一致。                         |
| op         | op           | 归约的操作类型。                                                       |
| group      | group        | 执行该操作的进程组实例。                                               |
| async_op   | sync_op      | 该操作是否为异步或同步操作，PyTorch 和 Paddle 取值相反，需要转写。 |

### 转写示例

#### async_op 参数：该操作是否为异步或同步操作

```python
# PyTorch 写法:
torch.distributed.reduce_scatter(data1, [data1, data2], async_op=False)

# Paddle 写法:
paddle.distributed.reduce_scatter(data1, [data1, data2], sync_op=True)
```
