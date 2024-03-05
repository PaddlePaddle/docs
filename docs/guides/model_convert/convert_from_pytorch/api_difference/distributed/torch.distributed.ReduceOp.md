## [torch 参数更多]torch.distributed.ReduceOp

### [torch.distributed.ReduceOp](https://pytorch.org/docs/stable/distributed.html?highlight=torch+distributed+reduceop#torch.distributed.ReduceOp)

```python
torch.distributed.ReduceOp
```

### [paddle.distributed.ReduceOp](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distributed/ReduceOp_cn.html)

```python
paddle.distributed.ReduceOp
```

两者功能一致。

其中，规约操作对应如下：

|  torch   | paddle  |
|  ----  | ----  |
| SUM  | SUM |
| PRODUCT  | PROD |
| MIN | MIN |
| MAX | MAX |
| BAND | - |
| BOR | - |
| BXOR | - |
| PREMUL_SUM | -  |
