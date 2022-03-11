Add instructions to reproduce your reported results.

pip install tensorboardX
pip install tqdm

------ testing ------

CIFAR-10:
------ labeled samples = 250 ------
filepath = "./Task2_logs_weights/weights_cifar10_250_latest/cifar10_VAT.pt"
python test.py --dataset cifar10 --num-labeled 250
------ labeled samples = 4000 ------
filepath = "./Task2_logs_weights/weights_cifar10_4k_latest/cifar10_VAT.pt"
python test.py --dataset cifar10 --num-labeled 4000
========================================================
CIFAR-100:
------ labeled samples = 2500 ------
filepath = "./Task2_logs_weights/weights_cifar100_2.5k_latest/cifar10_VAT.pt"
python test.py --dataset cifar100 --num-labeled 2500
------ labeled samples = 10000 ------
filepath = "./Task2_logs_weights/weights_cifar100_2.5k_latest/cifar10_VAT.pt"
python test.py --dataset cifar100 --num-labeled 10000