# metrics/classification_metrics.py
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def accuracy_score(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def precision(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return precision_score(labels, preds, average='binary')

def recall(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return recall_score(labels, preds, average='binary')

def f1score(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, preds, average='binary')

def auc_score(logits, labels):
    # get probabilities for positive class
    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
    labels = labels.cpu().numpy()
    return roc_auc_score(labels, probs)

def conf_matrix(logits, labels):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return confusion_matrix(labels, preds)

