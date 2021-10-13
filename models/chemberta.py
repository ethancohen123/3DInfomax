import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models

from transformers import RobertaForMaskedLM


class Mol_Encoder(torch.nn.Module):
    def __init__(self):
        super(Mol_Encoder, self).__init__()
        self.model_mol = RobertaForMaskedLM.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k") #pretrained encoder
        self.model_mol.lm_head=Identity()

    def forward(self, input_ids, attention_mask):
        output_1 = self.model_mol(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0,:]
       
        return pooler