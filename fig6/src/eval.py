# eval.py
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from wilds import get_dataset
from laplace import Laplace
from netcal.metrics import ECE
from sklearn.metrics import log_loss, accuracy_score
import numpy as np
from tqdm import tqdm

# 数据准备，与train.py保持一致
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
dataset = get_dataset('civilcomments', download=True)

class CCDataset(Dataset):
    def __init__(self, wilds_subset, tokenizer, max_length=128):
        self.samples = [wilds_subset[i] for i in range(len(wilds_subset))]
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __getitem__(self, idx):
        text = self.samples[idx]['text']
        label = self.samples[idx]['y']
        meta = self.samples[idx]['metadata']  # 8维身份特征
        enc = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        item['metadata'] = meta
        return item
    def __len__(self):
        return len(self.samples)

val_ds = CCDataset(dataset.get_subset('val'), tokenizer)
test_ds = CCDataset(dataset.get_subset('test'), tokenizer)

val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# 加载保存的模型
model = DistilBertForSequenceClassification.from_pretrained('./finetuned_distilbert_cc')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_probs_and_targets(model, dataloader):
    model.eval()
    all_probs, all_targets, all_metas = [], [], []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            probs = torch.softmax(outputs.logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(batch['labels'].cpu().numpy())
            all_metas.append(batch['metadata'].cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets), np.concatenate(all_metas)

# 评估 MAP
probs_map, targets, metas = get_probs_and_targets(model, test_loader)

# Laplace
la = Laplace(model, 'classification', subset_of_weights='last_layer', hessian_structure='full')
la.fit(val_loader)  # 也可用 train_loader
la.optimize_prior_precision(val_loader=val_loader)
probs_la, _, _ = get_probs_and_targets(la, test_loader)

# Groupwise metrics
def groupwise_metrics(probs, targets, metas):
    ece = ECE(15)
    group_ece, group_nll, group_acc = [], [], []
    for i in range(8):  # 8 groups
        mask = metas[:,i]==1
        if np.sum(mask)==0: continue
        group_ece.append(ece.measure(probs[mask], targets[mask]))
        group_nll.append(log_loss(targets[mask], probs[mask]))
        group_acc.append(accuracy_score(targets[mask], np.argmax(probs[mask], axis=1)))
    # “ID”是平均，“OOD”是最差
    return np.mean(group_ece), np.max(group_ece), np.mean(group_nll), np.max(group_nll), np.mean(group_acc), np.min(group_acc)

ece_id, ece_ood, nll_id, nll_ood, acc_id, acc_ood = groupwise_metrics(probs_map, targets, metas)
ece_id_la, ece_ood_la, nll_id_la, nll_ood_la, acc_id_la, acc_ood_la = groupwise_metrics(probs_la, targets, metas)

print(f"MAP:     NLL (ID)={nll_id:.3f}, ECE (ID)={ece_id:.3f}, Acc (ID)={acc_id:.3f}, NLL (OOD)={nll_ood:.3f}, ECE (OOD)={ece_ood:.3f}, Acc (OOD)={acc_ood:.3f}")
print(f"Laplace: NLL (ID)={nll_id_la:.3f}, ECE (ID)={ece_id_la:.3f}, Acc (ID)={acc_id_la:.3f}, NLL (OOD)={nll_ood_la:.3f}, ECE (OOD)={ece_ood_la:.3f}, Acc (OOD)={acc_ood_la:.3f}")
