import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
##############################################################
##############################################################
def getDataLoader(datasetForLoader, batchSize, shuffle, dropLast, numWorkers=8, sampler=None):
    myLoader = DataLoader(
        dataset=datasetForLoader,
        batch_size=batchSize,
        shuffle=shuffle,
        drop_last=dropLast,
        num_workers=numWorkers,
        sampler=sampler
    )
    return myLoader

def train_one_epoch(model, dataloader, optimizer, criterion, device, modelSave, rank):
    model.train()  # Set model to training mode (enables dropout)
    total_loss = 0
    correct_preds = 0
    total_samples = 0

    count = 0
    for batch in tqdm(dataloader, desc="Training"):
        count += 1
        # 1. Move data to GPU/CPU
        # Assuming the dataloader returns (input_ids, masks, labels)
        input_ids, masks, labels = [b.to(device) for b in batch]

        # 2. Clear previous gradients
        optimizer.zero_grad()

        # 3. Forward Pass
        logits, probs = model(input_ids, paddingMask=masks)

        # 4. Calculate Loss
        loss = criterion(logits, labels)

        # 5. Backward Pass (The "learning" part)
        loss.backward()

        # 6. Update Weights
        optimizer.step()

        # Statistics
        total_loss += loss.item()

        # Calculate Accuracy on the fly
        predictions = torch.argmax(logits, dim=-1)
        correct_preds += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / count
    accuracy = correct_preds / total_samples
    
    if rank == 0:
        checkpoint = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            # Optional: include hyperparameters or scheduler state
            }
        torch.save(checkpoint, modelSave)
    
    return avg_loss, accuracy
