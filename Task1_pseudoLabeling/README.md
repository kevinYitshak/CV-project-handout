Add instructions to reproduce your reported results.
If necessary create a requirements.txt of libraries that you additionally use. 

pip install tensorboardX
pip install tqdm

------ testing ------
CIFAR-10:
------ labeled samples = 250 ------
threshold: 0.6 => filepath = "./Task1_weights/weights_CIFAR10_0.6/cifar10.pt"
threshold: 0.75 => filepath = "./Task1_weights/weights_CIFAR10_0.75/cifar10.pt"
threshold: 0.95 => filepath = "./Task1_weights/weights_CIFAR10_0.95/cifar10.pt"

python test.py --dataset cifar10 --num-labeled 250
------ labeled samples = 4000 ------
threshold: 0.6 => filepath = "./Task1_weights/weights_CIFAR10_4k_0.6/cifar10.pt"
threshold: 0.75 => filepath = "./Task1_weights/weights_CIFAR10_4k_0.75/cifar10.pt"
threshold: 0.95 => filepath = "./Task1_weights/weights_CIFAR10_4k_0.95/cifar10.pt"

python test.py --dataset cifar10 --num-labeled 4000

CIFAR-100:
------ labeled samples = 2500 ------
threshold: 0.6 => filepath = "./Task1_weights/cifar100_2.5k_0.6_final/cifar10.pt"
threshold: 0.75 => filepath = "./Task1_weights/cifar100_2.5k_0.75_final/cifar10.pt"
threshold: 0.95 => filepath = "./Task1_weights/cifar100_2.5k_0.95_final/cifar10.pt"

python test.py --dataset cifar100 --num-labeled 2500
------ labeled samples = 10000 ------
threshold: 0.6 => filepath = "./Task1_weights/cifar100_10k_0.6_final/cifar10.pt"
threshold: 0.75 => filepath = "./Task1_weights/cifar100_10k_0.75_final/cifar10.pt"
threshold: 0.95 => filepath = "./Task1_weights/cifar100_10k_0.95_final/cifar10.pt"

python test.py --dataset cifar100 --num-labeled 10000