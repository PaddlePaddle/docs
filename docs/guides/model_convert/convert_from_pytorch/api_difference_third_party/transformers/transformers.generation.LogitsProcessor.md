## [输入参数用法不一致]transformers.generation.LogitsProcessor

### [transformers.generation.LogitsProcessor](https://hf-mirror.com/docs/transformers/v4.42.0/en/internal/generation_utils#logitsprocessor)

```python
transformers.generation.LogitsProcessor(input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs)
```

### [paddlenlp.generation.LogitsProcessor](https://github.com/PaddlePaddle/PaddleNLP/blob/e336e78c338d2514ee6c937982ce5d8c960b85ff/paddlenlp/generation/logits_process.py#L26)

```python
paddlenlp.generation.LogitsProcessor(input_ids: paddle.Tensor, scores: paddle.Tensor, **kwargs)
```

两者功能一致但参数不一致，部分参数名不同，具体如下：

### 参数映射

| PyTorch           | PaddlePaddle      | 备注                                                         |
| ----------------- | ----------------- | ------------------------------------------------------------ |
| input_ids         | input_ids         |  |
| scores            | logits            |  |
