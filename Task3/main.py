import argparse
from decimal import MAX_PREC
import math
from re import X
import numpy as np
import cv2 

from tqdm import tqdm

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy

from model.wrn  import WideResNet

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

def tensor2img(args, image):

    if args.num_classes == "10":
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2471, 0.2435, 0.2616])
    else:
        mean   = np.array([0.5071, 0.4867, 0.4408])
        std    = np.array([0.2675, 0.2565, 0.2761])

    tf = transforms.Compose([
        transforms.Normalize((-mean/std), (1/std))
    ])

    img = image.detach().cpu()
    img = tf(img).numpy()
    # print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    img = img * 255
    img = img.astype('uint8')
    return img

def img2tensor(args, image):

    if args.num_classes == "10":
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2471, 0.2435, 0.2616])
    else:
        mean   = np.array([0.5071, 0.4867, 0.4408])
        std    = np.array([0.2675, 0.2565, 0.2761])

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    return tf(image)

def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()
 
    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
 
    return normalized_cdf
 
def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table
 
def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)
 
    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0,256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0,256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0,256])    
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0,256])    
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0,256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0,256])
 
    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)
 
    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)
 
    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)
 
    # Put the image back together
    image_after_matching = cv2.merge([
        blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)
 
    return image_after_matching

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

    # print(labeled_dataset)
    
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
    # optim
    optim = torch.optim.SGD(
            model.parameters(),
            lr=1e-2,
            momentum=0.9,
            weight_decay=args.wd,
        )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=1e-2, steps_per_epoch=1024, epochs=10)
    ############################################################################
    
    best_acc = 0

    model.train()
    for epoch in range(args.epoch):

        print('-'*30)
        print("epoch: ", epoch)
        print('-'*30)
        train_loss_epoch = 0
        train_acc_epoch = 0
        iteration = 0
        # model.train()

        tbar = tqdm(range(args.iter_per_epoch))

        for i in tbar:
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
        
            selected_unlabeled_samples = 0

            # porb for unlabeled data
            y_ul_pred_prob = torch.nn.Softmax(dim=1)(y_ul_pred)
            # print(y_ul_pred_prob.size()) # [64, 10]

            max_pred_val, max_pred_label = torch.topk(y_ul_pred_prob, k=1, dim=-1)
            # print(max_pred_label) # lables
            # print(pred_val.shape, pred_idx.shape) # [64, 1]

            unlabel = torch.where(max_pred_val > 0.15, max_pred_label, 
                        (args.num_classes+1) * torch.ones(1, dtype=torch.long).to(device))

            # print(unlabel)
            # get index of unlable_index which is not = 11
            idx_unlabel, _ = torch.where(unlabel != args.num_classes+1)
            select_unlabel_samples = torch.index_select(x_ul, 0, torch.as_tensor(idx_unlabel, device=device))

            # print('selected_samples: ', select_unlabel_samples.size())

            unlabel_filtered = unlabel[unlabel != (args.num_classes+1)]

            if (unlabel_filtered.size() != 0):
                # label_samples = 0
                # label_output = 0
                unlabel_unique = torch.unique(unlabel_filtered)
                print('unlabel_unique: ', unlabel_unique.size(0))

                select_unlabel_unique = torch.empty(unlabel_unique.size(0), 3, 32, 32)
                select_label_samples = torch.empty(unlabel_unique.size(0), 3, 32, 32)
                select_label_output = torch.empty(unlabel_unique.size(0))

                for k in range(unlabel_unique.size(0)):
                    idx_unlabel_unique, _ = torch.where(unlabel == unlabel_unique[k])
                    select_unlabel_unique[k, :, :, :] = torch.index_select(x_ul, 0, torch.as_tensor(idx_unlabel_unique[0], device=device))
                
                    idx_label = torch.where(y_l == unlabel_unique[k])

                    select_label_samples[k, :, :, :] = torch.index_select(x_l, 0, idx_label[0][0])
                    select_label_output[k] = torch.index_select(y_l, 0, idx_label[0][0])

                # print('select_label_samples size: ', select_label_samples.size())
                # print('select_label_output size: ', select_label_output.size())
                
                # print('select_unlabel_output size: ', select_unlabel_samples.size())

                hist_matched_tensor = torch.empty(select_label_samples.size(0), 3, 32, 32)
                for i in range(select_unlabel_unique.size(0)): #max can args.num_classes
                    # print(i)
                    img_label = tensor2img(args, select_label_samples[i, :, :, :])
                    img_unlabel = tensor2img(args, select_unlabel_unique[i, :, :, :])
                    '---------------------------HIST MATCHING-----------------------------------'
                    image_label_matched = match_histograms(img_label, img_unlabel)
                    # cv2.imwrite('./img_matched.png', image_label_matched)
                    # cv2.imwrite('./img_label.png', img_label)
                    # cv2.imwrite('./img_unlable.png', img_unlabel)
                    '---------------------------HIST MATCHING-----------------------------------'
                    hist_matched_tensor[i, :, :, :] = img2tensor(args, image_label_matched)


            # print(hist_matched_tensor.size())
            unlabel_pred = model(select_unlabel_samples)

            hist_matched_pred = model(hist_matched_tensor.to(device=device))

            if (unlabel_filtered.size(0) != 0):
                unlabel_loss = criterion(unlabel_pred, unlabel_filtered)
                hist_loss = criterion(hist_matched_pred, select_label_output.long())
                final_loss = label_loss + unlabel_loss + hist_loss
            else:
                final_loss = label_loss
            
            final_loss.backward()
            # print("Loss: ", final_loss.item())
            iteration += 1
            train_loss_iter = final_loss.item()
            train_acc_iter = accuracy(y_l_pred, y_l)[0].item()

            tbar.set_description('loss: {:.4f}; acc: {:.4f}'.format(train_loss_iter, train_acc_iter))

            # writer.add_scalar('Train/Acc_iter', train_acc_iter, i)
            # writer.add_scalar('Train/Loss_iter', train_loss_iter, i)

            train_loss_epoch += train_loss_iter
            train_acc_epoch += train_acc_iter

            optim.step()
            scheduler.step()

        train_loss_epoch /= iteration
        train_acc_epoch /= iteration
        # print("Train Acc: ", train_acc_epoch)

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
        
        print("Test Acc: ", test_acc)

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
                        default=250, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.03, type=float, 
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.001, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=1024*10, type=int,
                        help='total number of iterations to run') # 512
    parser.add_argument('--iter-per-epoch', default=1024, type=int,
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