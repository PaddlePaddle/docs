## [torch 参数更多]torch.linalg.inv_ex

### [torch.linalg.inv_ex](https://pytorch.org/docs/stable/generated/torch.linalg.inv_ex.html)

```python
torch.linalg.inv_ex(A， *， check_errors=False, out=None)
```

### [paddle.linalg.inv](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/inv_cn.html)

```python
paddle.linalg.inv(x, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                                  |
| ------------ | ------------ | --------------------------------------------------------------------- |
| A            | x            | 输入 Tensor，仅参数名不一致。                                         |
| check_errors | -            | 是否检查错误，paddle 暂不支持                                         |
| out          | -            | 表示输出的 Tensor ， Paddle 无此参数，需要转写。                      |
| 返回值       | 返回值       | Pytorch 返回两个 out 与 info，Paddle 仅返回一个 Tensor：out，需转写。 |

### 转写示例

#### 返回值

```python
# PyTorch 写法:
torch.linalg.inv_ex(x)

# Paddle 写法:
## 注: 仅支持 check_errors=False 时的情况
(paddle.linalg.inv(x), paddle.zeros(x.shape[:-2], dtype='int64'))
```

#### out 参数：输出的 Tensor

```python
# PyTorch 写法:
torch.linalg.inv_ex(x, out=y)

# Paddle 写法:
paddle.assign((paddle.linalg.inv(x), paddle.zeros(x.shape[:-2], dtype='int64')), y)
```
