import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.hidden_fc_1(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.hidden_fc_2(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x
    
class SimpleMLPDeeper(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLPDeeper, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hidden_fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_fc_3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.hidden_fc_1(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.hidden_fc_2(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.hidden_fc_3(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x
        