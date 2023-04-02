## [ 组合替代实现 ] torch.cdist

### [torch.dist](https://pytorch.org/docs/stable/generated/torch.cdist.html?highlight=cdist#torch-cdist)

```python
torch.Tensor.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
```

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API转写。

### 转写示例
```python
# torch 写法
a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702]])
c = torch.cdist(a, b, p=2)

# paddle 写法
a = paddle.to_tensor(data=[[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 
    1.059]])
b = paddle.to_tensor(data=[[-2.1763, -0.4713], [-0.6986, 1.3702]])
sa, sb = a.shape, b.shape
# 四种情况 2d 和 3d data的笛卡尔积
if len(sa) == 2 and len(sb) == 2:
    x = paddle.empty(shape=(sa[0], sa[1]), dtype='float32')
    for i in range(sa[0]):
        for j in range(sb[0]):
            x[i, j] = paddle.dist(a[i], b[j], p=2)
elif len(sa) == 2 and len(sb) == 3:
    x = paddle.empty(shape=(sb[0], sa[0], sb[1]), dtype='float32')
    for i in range(sb[0]):
        for j in range(sa[0]):
            for k in range(sb[1]):
                x[i, j, k] = paddle.dist(a[j], b[i][k], p=2)
elif len(sa) == 3 and len(sb) == 2:
    x = paddle.empty(shape=(sa[0], sa[1], sb[0]), dtype='float32')
    for i in range(sa[0]):
        for j in range(sa[1]):
            for k in range(sb[0]):
                x[i, j, k] = paddle.dist(a[i][j], b[k], p=2)
else:
    x = paddle.empty(shape=(sa[0], sa[1], sb[1]), dtype='float32')
    for i in range(sa[0]):
        for j in range(sa[1]):
            for k in range(sb[1]):
                x[i, j, k] = paddle.dist(a[i, j], a[i, k], p=2)
c = x.clone()
```