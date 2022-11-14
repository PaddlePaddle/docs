import time
import paddle
import torch

a = paddle.to_tensor(1)
b = torch.tensor(1)
b.detach().numpy()
a.numpy()

# torch.cuda.set_device(2)

# a = torch.tensor(1,device=torch.device("cuda:2"))
# lis = []
# for i in range(2000):
#     lis.append(a)
# time.sleep(4)
    
# import numpy as np

# fake = np.array([0, 1, 1, 0]).astype(np.int64)
# fake = np.array(fake)
# fake1 = np.array([fake])
# np.save("fake_data.npy", fake)
# we = np.load('fake_data.npy')