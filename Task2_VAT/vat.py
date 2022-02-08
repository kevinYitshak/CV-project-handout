
import torch
import torch.nn as nn

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def kl_divergence(self, predictions, advPred):
        # measure how far the two distribution is
        '''
        p * log(p/q) = plog(p) - plog(q)

        predictions => [64, 10]
        advPred => [64, 10]
        '''
        # TODO returning NANs

        kl_dist = predictions * (torch.log(predictions) - torch.log(advPred))
        # print(kl_dist.size()) # [64, 10]
        kl_dist_ = kl_dist.mean(dim = (0, 1))
        return kl_dist_

    def forward(self, model, x):
        predictions = torch.softmax(model(x), dim=-1)
        advDistance = 0

        r = torch.randn(x.size(), requires_grad=True)
        r_norm  = torch.unsqueeze(torch.linalg.norm(r, dim=(0)), 0)
        # print(r_norm.grad.size())
        
        # TODO fix grad and check the graph is connected
        # read VAT paper
        for i in range(self.vat_iter):
            advExamples = x + self.xi * r_norm
            advPredictions = model(advExamples)
            advPredictions = torch.softmax(advPredictions, dim=-1)

            advDistance = self.kl_divergence(predictions, advPredictions)
            # advDistance.backward()
            
            # r = advDistance.grad
            # TODO try torch norm if this didn't work
            # r = torch.norm(r, dim=())
            # print(r.size())

        advPredictions = model(x + self.eps * r)
        loss = self.kl_divergence(predictions, advPredictions)
        return loss