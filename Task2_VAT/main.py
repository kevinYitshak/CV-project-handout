import argparse
import math
from tqdm import tqdm

from dataloader import get_cifar10, get_cifar100
from vat        import VATLoss
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
    criterion = torch.nn.CrossEntropyLoss()
#     optim = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optim = torch.optim.SGD(
            model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=args.wd,
        )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=1e-2, steps_per_epoch=1024, epochs=30)
    ############################################################################
    best_acc = 0
    
    model.train()
    for epoch in range(args.epoch):
        train_loss_epoch = 0
        train_acc_epoch = 0
        iteration = 0
        print('-'*30)
        print("epoch: ", epoch)
        print('-'*30)

        tbar = tqdm(range(args.iter_per_epoch))

        for i in tbar:
            
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
            # TODO: SUPPLY you code
            # zero the parameter gradients
            optim.zero_grad()
            
            train_loss_iter = 0
            train_acc_iter = 0
            # create obj for VATLoss class
            vatLoss = VATLoss(args, device)
            vaLoss = vatLoss.forward(model, x_ul)
            predictions = model.forward(x_l)            
            classificationLoss = criterion(predictions, y_l)
            loss = classificationLoss + args.alpha * vaLoss
            loss.backward()
#             print(i)
            # print("Loss: {0}, VATLoss: {1}".format(loss.item(), vaLoss.item()))
            optim.step()
            scheduler.step()
            
            iteration += 1
            train_loss_iter = loss.item()
            train_acc_iter = accuracy(predictions, y_l)[0].item()

            tbar.set_description('loss: {:.4f}; Vat: {:.4f}; acc: {:.4f}'.format(train_loss_iter,vaLoss.item(), train_acc_iter))

            # writer.add_scalar('Train/Acc_iter', train_acc_iter, i)
            # writer.add_scalar('Train/Loss_iter', train_loss_iter, i)

            train_loss_epoch += train_loss_iter
            train_acc_epoch += train_acc_iter

        train_loss_epoch /= iteration
        train_acc_epoch /= iteration

        with torch.no_grad():
            test_loss = 0
            test_acc = 0
            total = 0
            for i, (data, target) in enumerate(test_loader):
                model.eval()
                data, target = data.to(device), target.to(device)
                pred = model(data)
                loss = criterion(pred, target)

                test_loss += loss.item() / target.size(0)
                total += 1
                test_acc += accuracy(pred, target)[0].item()
        
        test_acc /= total
        test_loss /= total
        
        writer.add_scalar('Train/Acc', train_acc_epoch, epoch)
        writer.add_scalar('Train/Loss', train_loss_epoch, epoch)
        writer.add_scalar('Test/Acc', test_acc, epoch)
        writer.add_scalar('Test/loss', test_loss, epoch)
        
        print('Test Acc: ', test_acc)
        
        if (test_acc >= best_acc):
            best_acc = test_acc
            # torch.load('./weights/cifar10.pt')
            torch.save(model.state_dict(), './weights/cifar10_VAT.pt')
            ####################################################################



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=250, help='Total number of labeled samples')
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
    parser.add_argument('--total-iter', default=1024*30, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")                        
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--vat-xi", default=10.0, type=float, 
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=1.0, type=float, 
                        help="VAT epsilon parameter") 
    parser.add_argument("--vat-iter", default=1, type=int, 
                        help="VAT iteration parameter") 
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()

    main(args)