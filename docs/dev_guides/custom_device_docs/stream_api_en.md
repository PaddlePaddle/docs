# Stream APIs

## create_stream 【optional】

### Definition

```c++
C_Status (*create_stream)(const C_Device device, C_Stream* stream)
```

### Description

It creats a stream, which is used to execute asynchronous tasks within the framework. In the stream, tasks are done in order. When the device does not support asynchronous execution, the API is required to be implemented with an empty method.

### Parameter

device - the device to be used

stream - the created stream

## destroy_stream 【optional】

### Definition

```c++
C_Status (*destroy_stream)(const C_Device device, C_Stream stream)
```

### Description

It destroys a stream. When the device does not support asynchronous execution, the API needs to be implemented with an empty method.

### Parameter

device - the device to be used

stream - the stream required to be deallocated

## query_stream 【optional】

### Definition

```c++
C_Status (*query_stream)(const C_Device device, C_Stream stream)
```

### Description

It queries whether the tasks on the stream are done. If not implemented, it will be replaced with synchronize_stream by PaddlePaddle.

### Parameter

device - the device to be used

stream - the stream required to be queried.

## synchronize_stream 【optional】

### Definition

```c++
C_Status (*synchronize_stream)(const C_Device device, C_Stream stream)
```

### Description

It synchronizes the stream and waits for the completion of all tasks on the stream. When the device does not support asynchronous execution, the API is required to be implemented with an empty method.

### Parameter

device - the device to be used

stream - the stream needed to be synchronized

## stream_add_callback 【optional】

### Definition

```c++
C_Status (*stream_add_callback)(const C_Device device, C_Stream stream, C_Callback callback, void* user_data)
```

### Description

It adds a host callback function to the stream.

### Parameter

device - the device to be used

stream - the stream where the callback function is added

callback - the callback function

user_data - parameters of the function

## stream_wait_event 【optional】

### Definition

```c++
C_Status (*stream_wait_event)(const C_Device device, C_Stream stream, C_Event event)
```

### Description

It waits for the completion of an event on the stream. When the device does not support asynchronous execution, the API is required to be implemented with an empty method.

### Parameter

device - the device to be used

stream - the stream waited for

event - the event waited for
