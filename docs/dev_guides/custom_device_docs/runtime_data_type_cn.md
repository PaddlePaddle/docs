# 数据类型

## C_Status

### 类型定义

```c++
typedef enum {
  C_SUCCESS = 0,
  C_WARNING,
  C_FAILED,
  C_ERROR,
  C_INTERNAL_ERROR
} C_Status;
```

### 说明

C_SUCCESS - 函数执行成功时返回值。

C_WARNING - 函数功能可能不符合预期时返回值，例如异步接口实际是同步。

C_FAILED - 资源耗尽或请求失败。

C_ERROR - 参数错误，用法错误或未初始化。

C_INTERNAL_ERROR - 插件内部错误。

## C_Device

### 类型定义

```c++
typedef struct C_Device_st { int id; } * C_Device;
```

### 说明

描述一个 device 对象。

## C_Stream

### 类型定义

```c++
typedef struct C_Stream_st* C_Stream;
```

### 说明

描述一个 stream 对象，stream 是框架内部用于执行异步任务的任务队列，同一 stream 中的任务按顺序执行。

## C_Event

### 类型定义

```c++
typedef struct C_Event_st* C_Event;
```

### 说明

描述一个 event 对象，event 被框架内部用于同步不同 stream 之间的任务。

## C_Callback

### 类型定义

```c++
typedef void (*C_Callback)(C_Device device,
                           C_Stream stream,
                           void* user_data,
                           C_Status* status);
```

### 说明

主机回调函数类型，具有 4 个参数，使用的设备，使用的 stream，用户数据，以及返回值。

## CustomRuntimeParams

### 类型定义

```c++
struct CustomRuntimeParams {
  size_t size;
  C_DeviceInterface* interface;
  CustomRuntimeVersion version;
  char* device_type;
  char* sub_device_type;
  char reserved[32];
};
```

### 说明

插件入口函数 InitPlugin 的参数类型。

size - CustomRuntimeParams 的大小，框架和插件 CustomRuntimeParams 类型大小可能不一致，插件首先需要检查该大小，确保内存访问不会越界。可使用 PADDLE_CUSTOM_RUNTIME_CHECK_VERSION 宏完成检查。

interface - 设备回调接口，插件需要实现必要的接口，并填充该参数完成注册。

version - 使用 device_ext.h 头文件中定义的自定义 Runtime 版本填充，用于框架检查版本兼容性。

device_type - 设备类型名，用于框架区分设备，同时暴露到用户层，用于指定硬件后端，例如 "CustomCPU" 。

sub_device_type - 子设备类型名，可以用于说明插件版本，例如 "V1.0" 。

## CustomRuntimeVersion

### 类型定义

```c++
struct CustomRuntimeVersion {
  size_t major, minor, patch;
};
```

### 说明

插件使用的自定义 Runtime 的版本号，用于框架检查版本兼容性。可使用 PADDLE_CUSTOM_RUNTIME_CHECK_VERSION 宏完成填充。

## C_DeviceInterface

### 类型定义

C_DeviceInterface 的类型定义详见 [device_ext.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/device_ext.h)。

### 说明

自定义 Runtime 回调接口集合。

## C_CCLRootId

### 类型定义

C_CCLRootId 的类型定义详见 [device_ext.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/device_ext.h)。

### 说明

集合通讯根节点 unique_id 。

## C_CCLComm

### 类型定义

```c++
typedef struct C_CCLComm_st* C_CCLComm;
```

### 说明

集合通讯 communicator 数据结构。

## C_CCLReduceOp

### 类型定义

C_CCLReduceOp 的类型定义详见 [device_ext.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/device_ext.h)。

### 说明

集合通讯 Reduce 操作类型。
