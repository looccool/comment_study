import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, auc
)


def evaluate_model(model_name, data_loader, device):
    model_name.to(device)
    model_name.eval()  # Set to evaluation mode
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # batch is a tuple; 3rd element is label
            # Assuming: (input_ids, mask, labels)
            inputs, masks, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            # Forward pass: model returns (logits, probs)
            _, probs = model_name(inputs, paddingMask=masks)
            
            # For binary classification, we usually want the prob of the positive class (class 1)
            # If probs shape is (Batch, 2), we take the second column
            pos_probs = probs[:, 1].cpu().numpy()
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            
            all_probs.extend(pos_probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy for sklearn
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate Metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1_score": f1_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_probs)
    }
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)
    metrics['pr_auc'] = pr_auc

    # Store all raw scores in a DataFrame for detailed analysis
    results_df = pd.DataFrame({
        "ground_truth": all_labels,
        "predicted_class": all_preds,
        "probability_pos": all_probs
    })
    
    # Confusion Matrix (optional but very helpful)
    cm = confusion_matrix(all_labels, all_preds)
    metrics["confusion_matrix"] = cm

    return {
        "metrics": metrics,
        "predictions_df": results_df
    }
