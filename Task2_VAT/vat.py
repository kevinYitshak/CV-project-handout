
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import cv2

class VATLoss(nn.Module):

    def __init__(self, args, device):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter
        self.device = device
        self.args = args
    
    def forward(self, model, x):
        predictions = torch.softmax(model(x), dim=-1)
        
        noise = torch.normal(0, 1, size=x.shape).to(self.device)
        noise = self.L2Norm(noise)
        noise.requires_grad = True

        predictions = model(x)
    
        for _ in range(self.vat_iter):
            advPredictions = model(x + self.xi * noise)
            
            advDistance = F.kl_div(F.log_softmax(advPredictions, dim=-1), F.softmax(predictions, dim=-1)).backward(retain_graph=True)
            
            noise = self.L2Norm(noise.grad)
            model.zero_grad()

        image = x + self.eps * noise
        # to save some images for report
        # for i in range(x.size(0)-60):
        #     img = self.tensor2img(self.args, image[i, :, :, :])
        #     cv2.imwrite('./Task2_adv_Samples_' + str(i) + '.png', img)

        advOutput = model(image)

        loss = F.kl_div(F.log_softmax(advOutput, dim=-1), F.softmax(predictions, dim=-1))
        
        return loss
        

    def L2Norm(self, d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def tensor2img(self, args, image):

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
        print(img.shape)
        img = np.transpose(img, (1, 2, 0))
        img = img * 255
        img = img.astype('uint8')
        return img