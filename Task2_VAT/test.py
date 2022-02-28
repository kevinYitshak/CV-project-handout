import torch
import numpy as np
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
from utils import accuracy
import argparse
from torch.utils.data   import DataLoader

def test_cifar10(testdataset, device, filepath = ".weights/cifar10.pt"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 10]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    # TODO: SUPPLY the code for this function
    model = WideResNet(args.model_depth, 
                                10, widen_factor=args.model_width)
    model.load_state_dict(torch.load(filepath))
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    test_loss = 0
    total = 0

    for i, (data, target) in enumerate(testdataset):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = criterion(pred, target)

        test_loss += loss.item()
        total += target.size(0)
        acc = accuracy(pred, target)[0].item()
    
    return test_loss / total, acc / total

def test_cifar100(testdataset, device, filepath=".weights/cifar100.pt"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 100]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''
    model = WideResNet(args.model_depth, 
                                100, widen_factor=args.model_width)
    model.load_state_dict(torch.load(filepath))
    model.to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    test_loss = 0
    total = 0

    for i, (data, target) in enumerate(testdataset):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = criterion(pred, target)

        test_loss += loss.item()
        total += target.size(0)
        acc = accuracy(pred, target)[0].item()
    
    return test_loss / total, acc / total

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pseudo labeling \
                                        of CIFAR10/100 with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.00005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1*1, type=int,
                        help='total number of iterations to run') # 512
    parser.add_argument('--iter-per-epoch', default=1, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Confidence Threshold for pseudo labeling')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")

    args = parser.parse_args()
    args.num_classes = 10
    _, _, test_dataset = get_cifar10(args, args.datapath, mode="Test")

    test_loader = DataLoader(test_dataset,
                            batch_size = args.test_batch,
                            shuffle = False, 
                            num_workers=args.num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss, acc = test_cifar10(test_loader, device, "weights/cifar10.pt")
    print("Accuracy: {0}, Loss: {0}".format(acc, loss))