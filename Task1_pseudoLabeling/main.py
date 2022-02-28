import argparse
from decimal import MAX_PREC
import math
from re import X
import numpy as np

from sklearn.metrics import max_error

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy

from model.wrn  import WideResNet

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
from tensorboardX import SummaryWriter

def main(args):
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)

    print(labeled_dataset)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                    batch_size=args.train_batch,
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)
    
    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width)
    model       = model.to(device)

    ############################################################################
    # TODO: SUPPLY your code
    writer = SummaryWriter('./log')
    # optim
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # loss
    criterion = torch.nn.CrossEntropyLoss()
    ############################################################################
    
    best_acc = 0
    iteration = 0

    for epoch in range(args.epoch):

        print('-'*30)
        print("epoch: ", epoch)
        print('-'*30)
        train_loss_epoch = 0
        train_acc_epoch = 0
        # model.train()

        for i in range(args.iter_per_epoch):
            # ---------------------------
            # print("iter: ", i)
            # ---------------------------
            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
                x_ul, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul        = x_ul.to(device)
            ####################################################################
            # TODO: SUPPLY your code

            # print("Info about labelled and unlabelled data")
            # print("x_l: {0}, y_l: {1}".format(x_l.shape, y_l_onehot.size())) # [64, 3, 32, 32]; [64, 10]
            # print("x_ul: {0}, type: {1}".format(x_ul.shape, type(x_ul)))

            y_l_pred = model(x_l) # [64]
            y_ul_pred = model(x_ul)
            
            train_loss_iter = 0
            train_acc_iter = 0

            # zero the parameter gradients
            optim.zero_grad()

            # get loss for labeled data
            label_loss = criterion(y_l_pred, y_l)
            
            # start adding unlabeled data after training for some iterations
            if (epoch > 40):

                # porb for unlabeled data
                y_ul_pred_prob = torch.nn.Softmax(dim=1)(y_ul_pred)
                # print(y_ul_pred_prob)

                max_pred_val, max_pred_index = torch.max(y_ul_pred_prob, dim=-1, keepdim=True) # [64, 1]
                unlabel_index = torch.where(max_pred_val > args.threshold, max_pred_index, (args.num_class+1) * torch.ones(1, dtype=torch.long))
                
                # print(max_pred_val)
                # print(unlabel_index[unlabel_index != (args.num_class+1)])

                idx_numpy = torch.squeeze(unlabel_index, -1).numpy() # [64]
                
                # X_t = torch.cat((X_t, x_ul[idx_numpy[idx_numpy != 11]] ), dim=0) 
                selected_unlabeled_samples = x_ul[idx_numpy[idx_numpy != (args.num_class+1)]]

                # print(selected_unlabeled_samples.size())
            
                unlabel_pred = model(selected_unlabeled_samples)
                
                if (unlabel_index[unlabel_index != 11].size(0) != 0):
                    unlabel_loss = criterion(unlabel_pred, unlabel_index[unlabel_index != (args.num_class+1)])
                    final_loss = label_loss + unlabel_loss
                else:
                    final_loss = label_loss
            
            if ( epoch < 40):
                # print(final_loss)
                label_loss.backward()
                iteration += 1
                train_loss_iter = label_loss.item()
                train_acc_iter = accuracy(y_l_pred, y_l)[0].item()

                writer.add_scalar('Train/Acc_iter', train_acc_iter, i)
                writer.add_scalar('Train/Loss_iter', train_loss_iter, i)

                train_loss_epoch += train_loss_iter
                train_acc_epoch += train_acc_iter
            else:
                final_loss.backward()
                iteration += 1
                train_loss_iter = final_loss.item()
                train_acc_iter = accuracy(y_l_pred, y_l)[0].item()

                writer.add_scalar('Train/Acc_iter', train_acc_iter, i)
                writer.add_scalar('Train/Loss_iter', train_loss_iter, i)

                train_loss_epoch += train_loss_iter
                train_acc_epoch += train_acc_iter

            optim.step()

        train_loss_epoch /= iteration
        train_acc_epoch /= iteration

        with torch.no_grad():
            test_loss = 0
            total = 0
            for i, (data, target) in enumerate(test_loader):
                model.eval()
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = criterion(pred, target)

                test_loss += loss.item()
                total += target.size(0)
                acc = accuracy(pred, target)[0].item()
            
            test_acc = acc / total

        writer.add_scalar('Train/Acc', train_acc_epoch, epoch)
        writer.add_scalar('Train/Loss', train_loss_epoch, epoch)
        writer.add_scalar('Test/Acc', test_acc, epoch)
        writer.add_scalar('Test/loss', test_loss, epoch)

        if (test_acc > best_acc):
            best_acc = test_acc
            # torch.load('./weights/cifar10.pt')
            torch.save(model.state_dict(), './weights/cifar10.pt')
            ####################################################################
            



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
    
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)