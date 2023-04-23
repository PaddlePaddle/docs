## [ 仅参数名不一致 ] torch.bitwise_not

### [torch.bitwise_not](https://pytorch.org/docs/1.13/generated/torch.bitwise_not.html?highlight=bitwise_not#torch.bitwise_not)

```python
torch.bitwise_not(input,
                  *,
                  out=None) 
```

### [paddle.bitwise_not](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/bitwise_not_cn.html)

```python
paddle.bitwise_not(x,
                   out=None,
                   name=None)
```

两者功能一致且参数用法一致，仅参数名不同，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| <font color='red'>input</font>         | <font color='red'>x</font>            | 输入的 Tensor ，仅参数名不一致。                                      |
| <font color='red'>out</font>           | <font color='red'>out</font>             | 表示输出的 Tensor，参数名一致。               |

