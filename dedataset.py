import os
import random
import numpy as np
base_path = "./data/FB15K237/"
with open(os.path.join(base_path, "train2id.txt"), 'r') as f1, open(os.path.join(base_path, "train2id60.txt"), 'w') as f2:
    data = f1.readlines()
    print(data[0])
    cnt = np.int(0.6 * int(data[0]))
    print(cnt)
    f2.write(str(cnt))
    f2.write('\n')
    for i in range(cnt):
        f2.write(data[i + 1])


