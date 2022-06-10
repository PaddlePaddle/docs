# Event APIs

## create_event 【required】

### Definition

```c++
C_Status (*create_event)(const C_Device device, C_Event* event)
```

### Description

It creates an event, which is used to synchronize tasks of different streams within the framework. When the device does not support asynchronous execution, empty implementation of the API is required.

### Paremeter

device - the device to be used

event - the created event in storage

## destroy_event 【required】

### Definition

```c++
C_Status (*destroy_event)(const C_Device device, C_Event event)
```

### Description

It destroys an event. When the device does not support asynchronous execution, the API requires an empty implementation.

### Parameter

device - the device to be used

event - the event needed to be destroyed

## record_event 【required】

### Definition

```c++
C_Status (*record_event)(const C_Device device, C_Stream stream, C_Event event)
```

### Description

It records the event on the stream. When the device does not support asynchronous execution, empty implementation of the API is required.

### Parameter

device - the device to be used

stream - the stream where the event is recorded

event - the recorded event

## query_event 【optional】

### Definition

```c++
C_Status (*query_event)(const C_Device device, C_Event event)
```

### Description

It queries whether the event is complete. If not implemented, PaddlePaddle will use synchronize_event instead.

### Parameter

device - the device to be used

event - the event to be queried

## synchronize_event 【required】

### Definition

```c++
C_Status (*synchronize_event)(const C_Device device, C_Event event)
```

### Description

It synchronizes the event and waits for its completion. When the device does not support asynchronous execution, empty implementation of the API is required.

### Parameter

device - the device to be used

event - the event required to be synchronized
