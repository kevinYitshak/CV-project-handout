
import torch
import torch.nn as nn

class VATLoss(nn.Module):

    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def forward(self, model, x):
        predictions = torch.softmax(model(x), dim=-1)
        advDistance = 0

        r = torch.randn(x.size(), requires_grad=True)
        r_norm  = torch.unsqueeze(torch.linalg.norm(r, dim=(0)), 0)
        # print(r_norm.grad.size())

        for i in range(self.vat_iter):
            advExamples = x + self.xi * r_norm
            advPredictions = model(advExamples)
            advPredictions = torch.softmax(advPredictions, dim=-1)

            # advDistance = self.kl_divergence(predictions, advPredictions)
            advDistance = nn.functional.kl_div(nn.functional.log_softmax(advPredictions), nn.functional.softmax(predictions))
            advDistance.backward(retain_graph = True)
            
            r = torch.unsqueeze(torch.linalg.norm(r.grad, dim=(0)), 0)
            model.zero_grad()
            
        advPredictions = model(x + self.eps * r)
        loss = nn.functional.kl_div(nn.functional.log_softmax(advPredictions), nn.functional.softmax(predictions))
        return loss