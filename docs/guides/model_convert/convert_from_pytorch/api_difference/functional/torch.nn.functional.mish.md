## [ torch 参数更多 ]torch.nn.functional.mish

### [torch.nn.functional.mish](https://pytorch.org/docs/1.13/generated/torch.nn.functional.mish.html?highlight=torch+nn+functional+mish#torch.nn.functional.mish)

```python
torch.nn.functional.mish(input,
                         inplace=False)
```

### [paddle.nn.functional.mish](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/mish_cn.html)

```python
paddle.nn.functional.mish(x,
                         name=None)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x         | 表示输入的 Tensor ，仅参数名不一致。                                     |
| inplace     | -            | 表示在不更改变量的内存地址的情况下，直接修改变量的值，Paddle 无此参数，需要进行转写。 |

### 转写示例

#### inplcae

```python
# PyTorch 写法
y = torch.nn.functional.mish(x, True)

# Paddle 写法
y = paddle.nn.functional.mish(x)
paddle.assign(y, x)
```
