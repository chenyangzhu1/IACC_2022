修改 LAS_AWP_train_cifar10.py 中的第13行
加入import




RuntimeError: CUDA out of memory. Tried to allocate 80.00 MiB
(GPU 0; 10.75 GiB total capacity; 9.11 GiB already allocated;
81.56 MiB free; 9.38 GiB reserved in total by PyTorch)
If reserved memory is >> allocated memory try setting max_split_size_mb
to avoid fragmentation.  See documentation for Memory
Management and PYTORCH_CUDA_ALLOC_CONF

在LAS_AWP/CIFAR10_models/wide_resnet.py 第31行加入
if hasattr(torch.cuda, 'empty_cache'):
	torch.cuda.empty_cache()
没用

LAS_AWP/LAS_AWP_train_cifar10.py
第254，255行修改batch_size 为64
有效


LAS_AWP/LAS_AWP_train_cifar10.py
257行 200->10

LAS_AWP/LAS_AWP_train_cifar10.py
861行
if(i>=10):
    break
让程序早点退出

LAS_AWP/LAS_AWP_train_cifar10.py
894行
加入break
让程序早点退出

LAS_AWP/LAS_AWP_train_cifar100.py
import 类似cifar10

LAS_AWP/LAS_AWP_train_cifar10.py 成功运行
LAS_AWP/LAS_AWP_train_cifar100.py 炸梯度？
LAS_AWP/LAS_AWP_train_TinyImageNet.py 成功运行

LAS_PGD_AT/LAS_AT_train_cifar10.py 成功运行

LAS_PGD_AT/LAS_AT_train_ImageNet.py

36行改为wideresnet  没用 改回

LAS_PGD_AT/TinyImageNet_models/preact_resnet.py
95行添加resize

ImageNet_models/wide_resnet.py
修改种类数为1000

LAS_PGD_AT/cifar10改.py
94行 修改种类数为1000
```
    one_hot = np.eye(1000)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(1000 - 1))

```

修改interval 快速验证
train加入break  验证test
