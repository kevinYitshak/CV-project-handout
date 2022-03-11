Add instructions to reproduce your reported results.

pip install tensorboardX
pip install tqdm
pip install opencv-python

------ testing for our implementation ------

CIFAR-10:
------ labeled samples = 250 ------
filepath = "./Task3_logs/CIFAR10_250_ours/ckpt/cifar10.pth.tar"
python test.py --dataset cifar10 --num-labeled 250

------ labeled samples = 4000 ------
filepath = "./Task3_logs/CIFAR10_4k_ours/ckpt/cifar10.pth.tar"
python test.py --dataset cifar10 --num-labeled 4000

------ testing for our implementation of fixmatch ------
CIFAR-10:
------ labeled samples = 2500 ------
filepath = "./Task3_logs/CIFAR10_250_fixmatch_only/ckpt/cifar10.pth.tar"
python test.py --dataset cifar10 --num-labeled 250

------ labeled samples = 4000 ------
filepath = "./Task3_logs/CIFAR10_4k_fixmatch_only/ckpt/cifar10.pth.tar"
python test.py --dataset cifar10 --num-labeled 4000