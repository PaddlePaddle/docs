## [ 返回参数类型不一致 ]torch.equal
### [torch.equal](https://pytorch.org/docs/stable/generated/torch.equal.html?highlight=equal#torch.equal)

```python
torch.equal(input,
            other)
```

### [paddle.equal_all](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/equal_all_cn.html#equal-all)

```python
paddle.equal_all(x,
                 y,
                 name=None)
```

两者功能一致但返回参数类型不同，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'> input </font> | <font color='red'> x </font> | 表示输入的 Tensor ，仅参数名不一致。  |
| <font color='red'> other </font> | <font color='red'> y </font> | 表示输入的 Tensor ，仅参数名不一致。  |

注：Pytorch 返回 bool 类型，Paddle 返回 0-D bool Tensor


### 转写示例
#### 返回值
``` python
# Pytorch 写法
out = torch.equal(x, y)

# Paddle 写法
out = paddle.equal_all(x, y)
out = out.item()
```
