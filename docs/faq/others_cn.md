# 其他常见问题


##### 问题：使用 X2paddle 从 Caffe 转 Paddle model 时，报错 `TypeError: __new__() got an unexpected keyword argument 'serialized_options'` ，如何处理？

+ 答复：这是由于 ProtoBuf 版本较低导致，将 protobuf 升级到 3.6.0 即可解决。

----------

##### 问题：Windows 环境下，出现"Windows not support stack backtrace yet"，如何处理？

+ 答复：Windows 环境下，遇到程序报错不会详细跟踪内存报错内容。这些信息对底层开发者更有帮助，普通开发者不必关心这类警告。如果想得到完整内存追踪错误信息，可以尝试更换至 Linux 系统。

----------
