## [ 参数不一致 ]torch.sub
### [torch.sub](https://pytorch.org/docs/stable/generated/torch.sub.html?highlight=torch%20sub#torch.sub)

```python
torch.sub(input,
          other,
          *,
          alpha=1,
          out=None)
```

### [paddle.subtract](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/subtract_cn.html#subtract)

```python
paddle.subtract(x,
                y,
                name=None)
```

Pytorch 的 `other` 参数与 Paddle 的 `y` 参数用法不同，且支持更多参数，功能有较大差异，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 表示被减数的 Tensor，仅参数名不一致。  |
| other         | y            | 表示减数。Pytorch 中可以为 Tensor 或者 number，Paddle 中仅为 Tensor。此外，Pytorch 中的 other 可与 alpha 相乘成为最终的减数，PaddlePaddle 无此功能。  |
| alpha         | -            | 表示`other`的乘数，PaddlePaddle 无此参数。  |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。  |


### 功能差异
**PyTorch**：从`input`中减去和`alpha`相乘的`other`。

**PaddlePaddle**：逐元素相减算子，输入` x` 与输入 `y` 逐元素相减，并将各个位置的输出元素保存到返回结果中。

#### 计算差异
***PyTorch***：
$ out = input - alpha * other $

***PaddlePaddle***：
$ out = x - y $
