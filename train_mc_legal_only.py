"""
Train a neural network to pick the best card among only the legal cards in hand, using MC oracle data.
No reference to RuleBased agent. Only MC data is used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List
from src.schafkopf_ai.env import CARD_TO_IDX, IDX_TO_CARD

class LegalMCDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.samples = []
        for _, row in df.iterrows():
            hand = str(row['hand']).split('|') if pd.notna(row['hand']) and row['hand'] else []
            legal = str(row['legal_actions']).split('|') if pd.notna(row['legal_actions']) and row['legal_actions'] else []
            best = row['best_action']
            # Only train if best action is legal and in hand
            if best not in legal or best not in hand:
                continue
            # Features: one-hot for each card in hand (max 8)
            hand_vec = [CARD_TO_IDX[c] for c in hand if c in CARD_TO_IDX]
            legal_vec = [CARD_TO_IDX[c] for c in legal if c in CARD_TO_IDX]
            best_idx = legal_vec.index(CARD_TO_IDX[best])
            # Game context features (trick number, is_declarer, player_idx)
            trick_number = float(row['trick_number']) / 7.0
            is_declarer = float(row['is_declarer'])
            player_idx = float(row['player_idx']) / 3.0
            context = np.array([trick_number, is_declarer, player_idx], dtype=np.float32)
            self.samples.append((hand_vec, legal_vec, best_idx, context))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        hand_vec, legal_vec, best_idx, context = self.samples[idx]
        # Pad legal_vec to 8
        padded = np.full(8, -1, dtype=np.int64)
        padded[:len(legal_vec)] = legal_vec
        mask = np.zeros(8, dtype=np.float32)
        mask[:len(legal_vec)] = 1.0
        return {
            'legal': torch.tensor(padded),
            'mask': torch.tensor(mask),
            'best': torch.tensor(best_idx),
            'context': torch.tensor(context),
        }

class LegalMCNet(nn.Module):
    def __init__(self, context_dim=3, card_embed_dim=16, hidden=64):
        super().__init__()
        self.card_embed = nn.Embedding(32, card_embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(8 * card_embed_dim + context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 8)
        )
    def forward(self, legal, context):
        # legal: (B, 8) indices, context: (B, 3)
        emb = self.card_embed(legal.clamp(min=0))  # (B, 8, D)
        emb[legal == -1] = 0  # Mask out padded
        flat = emb.view(emb.size(0), -1)
        x = torch.cat([flat, context], dim=1)
        logits = self.fc(x)
        return logits

def collate(batch):
    legal = torch.stack([b['legal'] for b in batch])
    mask = torch.stack([b['mask'] for b in batch])
    best = torch.stack([b['best'] for b in batch])
    context = torch.stack([b['context'] for b in batch])
    return {'legal': legal, 'mask': mask, 'best': best, 'context': context}

def train():
    ds = LegalMCDataset('data/mc_training_data.csv')
    dl = DataLoader(ds, batch_size=128, shuffle=True, collate_fn=collate)
    net = LegalMCNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(200):
        net.train()
        total, correct = 0, 0
        for batch in dl:
            logits = net(batch['legal'], batch['context'])
            logits = logits + (batch['mask'] - 1) * 1e9  # Mask out padded
            loss = F.cross_entropy(logits, batch['best'])
            opt.zero_grad()
            loss.backward()
            opt.step()
            pred = logits.argmax(dim=1)
            correct += (pred == batch['best']).sum().item()
            total += len(pred)
        print(f"Epoch {epoch+1}: acc={correct/total:.3f}")
    torch.save(net.state_dict(), 'checkpoints/legal_mc_agent.pt')

if __name__ == '__main__':
    train()
