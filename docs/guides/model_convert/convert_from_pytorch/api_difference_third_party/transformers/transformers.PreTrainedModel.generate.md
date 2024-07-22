## [返回参数类型不一致]transformers.PreTrainedModel.generate

### [transformers.PreTrainedModel.generate](https://github.com/huggingface/transformers/blob/0fdea8607d7e01eb0e38a1ebeb7feee30a22f0cf/src/transformers/generation/utils.py#L1567)

```python
transformers.PreTrainedTokenizer.encode(input, **kwargs)
```

### [paddlenlp.transformers.PreTrainedModel.generate](https://github.com/PaddlePaddle/PaddleNLP/blob/88d4b19bc6865fb28c11d2ce83d07c3b4b8dc423/paddlenlp/generation/utils.py#L604)

```python
paddlenlp.transformers.PreTrainedTokenizer.encode(input_ids, **kwargs)
```

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | input_ids    | token 的 id 组成的 Tensor。 |
| 返回值         | 返回值        | PyTorch 返回的 Tensor 包含输入的 Tensor，Paddle 不包含且返回类型为 tuple。|

### 转写示例

```python
# Pytorch 写法
res = transformers.PreTrainedModel.generate(input = input_x)

# Paddle 写法
temp_res = paddlenlp.transformers.PreTrainedModel.generate(input_ids = input_x)
res = paddle.concat((input_x,temp_res[0]),axis=-1)
```
