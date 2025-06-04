# --- src/train.py ------------------------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizerFast,
                          DistilBertForSequenceClassification,
                          Trainer, TrainingArguments,
                          DataCollatorWithPadding)
from wilds import get_dataset

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

raw_ds = get_dataset('civilcomments', download=True)

class CCDataset(Dataset):
    def __init__(self, wilds_subset, tokenizer, max_len=128):
        self.dset = wilds_subset          # wilds Subset object
        self.tok  = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        text, y, meta = self.dset[idx]   # ⬅️ 解包 tuple
        enc = self.tok(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"]    = torch.tensor(y, dtype=torch.long)
        item["metadata"]  = meta           # already torch tensor
        return item
# -----------------------------------------------------------

train_ds = CCDataset(raw_ds.get_subset('train'), tokenizer)
val_ds   = CCDataset(raw_ds.get_subset('val'), tokenizer)

data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    output_dir          = "ckpt",
    num_train_epochs    = 2,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    evaluation_strategy = "epoch",
    save_strategy       = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    fp16 = torch.cuda.is_available(),
)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)

trainer = Trainer(
    model          = model,
    args           = training_args,
    train_dataset  = train_ds,
    eval_dataset   = val_ds,
    data_collator  = data_collator,
)

trainer.train()

# 保存模型
trainer.save_model("../model/finetuned_distilbert_cc")
print("训练完成，模型已保存到 ../model/finetuned_distilbert_cc")
