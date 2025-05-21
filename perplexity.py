import torch
from torcheval.metrics.text import Perplexity
from tqdm import tqdm


def calc_perplexity(model, train_dataloader, ignore_index):
    metric = Perplexity(ignore_index=ignore_index).to("cuda")
    model = model.to("cuda")
    for batch in tqdm(
        train_dataloader, desc="Computing ppl", total=len(train_dataloader)
    ):
        input_ids, attention_mask = batch
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        with torch.no_grad():
            logits = model(input_ids, attention_mask)[:, :-1]
        targets = input_ids[:, 1:]
        metric.update(logits, targets)
    ppl = metric.compute().item()
    return ppl
