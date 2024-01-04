## [ torch 参数更多 ]torch.msort

### [torch.msort](https://pytorch.org/docs/stable/generated/torch.msort.html#torch.msort)

```python
torch.msort(input, *, out=None)
```

### [paddle.sort](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sort_cn.html#sort)

```python
paddle.sort(x, axis=- 1, descending=False, name=None)
```

其中 PyTorch 与 Paddle 有不同的参数，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示输入的 Tensor ，仅参数名不一致。                   |
| -         | axis            | 排序的维度，当维度为 0 时，Paddle 与 PyTorch 功能一致。                  |
| -         | descending            | 设置是否降序排列。PyTorch 无此参数，Paddle 保持默认即可。                  |
| out         | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写 。                   |

### 转写示例
#### out：表示输出的 Tensor
```python
# PyTorch 写法
torch.msort(input, out=out)

# Paddle 写法
paddle.assign(paddl.sort(input, axis=0), out)
```
