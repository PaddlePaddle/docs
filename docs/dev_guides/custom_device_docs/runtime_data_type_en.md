# Data Type

## C_Status

### Definition

```c++
typedef enum {
  C_SUCCESS = 0,
  C_WARNING,
  C_FAILED,
  C_ERROR,
  C_INTERNAL_ERROR
} C_Status;
```

### Description

C_SUCCESS - The returned value when the execution of the function is a success

C_WARNING - The returned value when the performance of the funtion falls short of expectations. For example, the asynchronous API is actually synchronous.

C_FAILED - Resources runs out or the request fails.

C_ERROR - Parameter error, incorrect usage, or not initialized.

C_INTERNAL_ERROR - Plug-in internal error

## C_Device

### Definition

```c++
typedef struct C_Device_st { int id; } * C_Device;
```

### Description

It describes a device.

## C_Stream

### Definition

```c++
typedef struct C_Stream_st* C_Stream;
```

### Description

It describes a stream, which is used to execute asynchronous tasks within the framework. In the stream, tasks are executed in order.

## C_Event

### Definition

```c++
typedef struct C_Event_st* C_Event;
```

### Description

It describes an event, which is used to synchronize tasks from different streams within the framework.

## C_Callback

### Definition

```c++
typedef void (*C_Callback)(C_Device device,
                           C_Stream stream,
                           void* user_data,
                           C_Status* status);
```

### Description

It is the callback function offered by the host and has four parameters: device, stream, user data, and returned value.

## CustomRuntimeParams

### Definition

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

### Description

They are function parameters of InitPlugin.

size - the size of CustomRuntimeParams. The size of the framework and the plug-in may be different. You need to first check the size of the plug-in and ensure that memory access does not cross the boundary. It is feasible to use the macro of PADDLE_CUSTOM_RUNTIME_CHECK_VERSION in the check.

interface - the device callback interface. It is necessary for the plug-in to implement essential APIs and fill the parameter in to finish registration.

version - the custom runtime version defined in the device_ext.h, which is used to check the version compatibility by the framework.

device_type - the appellation of the device type, used by the framework to distinguish devices and exposed to the user layer to specify the hardware back end, such as "CustomCPU".

sub_device_type - the appellation of the sub-device type, used to interpret the plug-in version, such as "V1.0".

## CustomRuntimeVersion

### Definition

```c++
struct CustomRuntimeVersion {
  size_t major, minor, patch;
};
```

### Description

It is the custom runtime version used by the plug-in. It is used to check the version compatibility by the framework and can be filled up by the macro of PADDLE_CUSTOM_RUNTIME_CHECK_VERSION.

## C_DeviceInterface

### Definition

For detailed definitions of the types of C_DeviceInterface, please refer to [device_ext.h](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/backends/device_ext.h).

### Description

It collects the custom runtime callback APIs.
