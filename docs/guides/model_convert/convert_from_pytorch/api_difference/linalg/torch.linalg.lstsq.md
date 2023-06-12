## [ 参数不一致 ]torch.linalg.lstsq
### [torch.linalg.lstsq](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html?highlight=lstsq#torch.linalg.lstsq)

```python
torch.linalg.lstsq(A, B, rcond=None, *, driver=None)
```

### [paddle.linalg.lstsq](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/lstsq_cn.html)

```python
paddle.linalg.lstsq(x, y, rcond=None, driver=None, name=None)
```

其中 PyTorch 与 Paddle 的返回值类型不一致，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| A         | x         | 表示输入的 Tensor 。                                     |
| B           | y           | 表示输入的 Tensor 。     |
| rcond           | rcond           | 用来决定 x 有效秩的 float 型浮点数。               |
| driver           | driver           | 用来指定计算使用的 LAPACK 库方法。               |
| return     | return            | 表示返回值，PyTorch 第三个返回值的类型为 int64，Paddle 第三个返回值的类型为 int32 ， 需要进行转写。 |

### 转写示例

#### return：返回值类型

```python
# Pytorch 写法
result = torch.linalg.lstsq(x, y, driver="gels")

# Paddle 写法
solution, residuals, rank, singular_values = paddle.linalg.lstsq(x=x, y=y，driver='gels')
rank = rank.astype('int64')
```
