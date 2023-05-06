## [ torch 参数更多]torch.tril_indices

### [torch.tril_indices](https://pytorch.org/docs/1.13/generated/torch.tril_indices.html?highlight=tril_indices#torch.tril_indices)

```python
torch.tril_indices(row,col,offset=0,*,dtype=torch.long,device='cpu',layout=torch.strided)
```

### [paddle.tril_indices](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/tril_indices_cn.html)

```python
paddle.tril_indices(row,col,offset=0,dtype='int64')
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射
| PyTorch | PaddlePaddle | 备注 |
| ------- | ------- | ------- |
| row | row | 表示输入 Tensor 的行数。 |
| col | col | 表示输入 Tensor 的列数。 |
| offset | offset | 表示从指定二维平面中获取对角线的位置。如果 offset = 0 ，取主对角线；如果 offset > 0 ，取主对角线右上的对角线；如果 offset < 0 ，取主对角线左下的对角线。 |
| dtype | dtype | 表示输出张量的数据类型。 |
| device | - | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。 |
| layout | - | 表示布局方式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

#### device: Tensor 的设备

```python
# Pytorch 写法
torch.tril_indices(row,col,offset,dtype,device=torch.device('cpu'))

# Paddle 写法
y = paddle.tril_indices(row,col,offset,dtype)
y.cpu()
```
