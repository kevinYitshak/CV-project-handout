
import torch
import torch.nn as nn

class VATLoss(nn.Module):

    def __init__(self, args, device):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter
        self.device = device

    def forward(self, model, x):
        predictions = torch.softmax(model(x), dim=-1)
        advDistance = 0

        r = torch.randn_like(x)
        
        for _ in range(self.vat_iter):
            r = self.xi * self.get_normalized_vector(r).requires_grad_().to(self.device)
            advExamples = x + r
            advPredictions = model(advExamples)
            advPredictions = torch.softmax(advPredictions, dim=-1)

            # advDistance = self.kl_divergence(predictions, advPredictions)
            advDistance = nn.functional.kl_div(nn.functional.log_softmax(advPredictions,dim=-1), nn.functional.softmax(predictions, dim=-1), reduction='batchmean')
#             advDistance.backward(retain_graph = True)
            
            grad = torch.autograd.grad(advDistance, [r])[0]
            r = grad.detach()
            r = self.get_normalized_vector(r).to(self.device)
          
#             model.zero_grad()
            
        advPredictions = model(x + self.eps * r)
        loss = nn.functional.kl_div(nn.functional.log_softmax(advPredictions, dim=-1), nn.functional.softmax(predictions, dim=-1), reduction='batchmean')
        return loss
    
    def get_normalized_vector(self, d):
        d_abs_max = torch.max(
            torch.abs(d.view(d.size(0), -1)), 1, keepdim=True)[0].view(
                d.size(0), 1, 1, 1)
        # print(d_abs_max.size())
        d = d / (1e-12 + d_abs_max)
        d = d / torch.sqrt(1e-6 + torch.sum(
            torch.pow(d, 2.0), tuple(range(1, len(d.size()))), keepdim=True))
        # print(torch.norm(d.view(d.size(0), -1), dim=1))
        return d