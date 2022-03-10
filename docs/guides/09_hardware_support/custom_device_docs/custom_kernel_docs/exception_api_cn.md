# Exception API

Exception API方便异常判断，具体参照[enforce.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/enforce.h)与[errors](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/core/errors.h)：

基本使用方式为：
```
 PADDLE_ENFORCE_TYPE(cond_a, // 条件A
                     cond_b, // 条件B, 根据TYPE可选
                     phi::errors::ERR_TYPE("ERR_MSG"));
```

其中根据`TYPE`的不同，支持的判断包括：

- `PADDLE_ENFORCE_EQ`：cond_a == cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_NE`：cond_a != cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_GT`：cond_a > cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_GE`：cond_a >= cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_LT`：cond_a < cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_LE`：cond_a <= cond_b，否则触发ERR_TYPE异常和报ERR_MSG
- `PADDLE_ENFORCE_NOT_NULL`：cond_a != nullptr，否则触发ERR_TYPE异常和报ERR_MSG

其中配合使用的ERR_TYPE支持包括：

- `InvalidArgument`：非法参数
- `NotFound`：未找到
- `OutOfRange`：越界
- `AlreadyExists`：已存在
- `ResourceExhausted`：资源超限
- `PreconditionNotMet`：前置条件未满足
- `PermissionDenied`：权限限制
- `ExecutionTimeout`：超时
- `Unimplemented`：未实现
- `Unavailable`：不可用
- `Fatal`：Fatal错误
- `External`：外部错误

其中ERR_MSG为C语言风格字符串，支持变长参数。

Exception API使用举例如下：

```c++
PADDLE_ENFORCE_EQ(
      (num_col_dims >= 2 && num_col_dims <= src.size()),
      true,
      phi::errors::InvalidArgument("The num_col_dims should be inside [2, %d] "
                                   "in flatten_to_3d, but received %d.",
                                   src.size(),
                                   num_col_dims));
```
当num_col_dims >= 2 && num_col_dims <= src.size()不为true时，报非法参数错误并输出报错信息。

>注：Kernel函数实现可用的飞桨开放API众多，无法一一列出，但框架内外使用一致，更详细的API用法请按需参照相应头文件和[飞桨框架](https://github.com/PaddlePaddle/Paddle)内的使用
