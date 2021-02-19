from torch import nn


class DisModel(nn.Module):
    def __init__(self, hidden_dim, label_dim):
        super(DisModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim

        # D (consistency adversarial loss)
        self.feature_dis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, 1, bias=True)
        )

        # C (consistency classification)
        self.consistency_dis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 8, bias=True),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 8, 3, bias=True)
        )

        # classification
        self.classifier = nn.ModuleDict({
            'img': nn.Sequential(
                nn.Linear(hidden_dim, label_dim, bias=True),
                nn.Sigmoid()
            ),
            'txt': nn.Sequential(
                nn.Linear(hidden_dim, label_dim, bias=True),
                nn.Sigmoid()
            ),
        })

    def dis_D(self, f):
        score = self.feature_dis(f)
        return score.squeeze()

    def dis_C(self, f):
        res = self.consistency_dis(f)
        return res.squeeze()

    def dis_classify(self, h, modality):
        res = self.classifier[modality](h)
        return res.squeeze()
