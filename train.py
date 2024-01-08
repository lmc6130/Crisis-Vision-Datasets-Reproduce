import numpy as np
import time
import torch
import wandb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix


def train_one_epoch(model, device, criterion, optimizer, train_data_loader):
    epoch_loss = []
    epoch_acc = []
    epoch_ps = []
    epoch_rs = []
    epoch_f1 = []
    trues = []
    prediction = []
    start_time = time.time()

    model.train()

    for batch_idx, (images, labels) in enumerate (train_data_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, labels)

        # Calculating Loss
        epoch_loss.append(loss.item())

        # Calculating Metrics
        _, predicts = preds.max(1)
        predicts = predicts.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        acc = accuracy_score(labels, predicts)
        ps = precision_score(labels, predicts, average="weighted")
        rs = recall_score(labels, predicts, average="weighted")
        f1 = f1_score(labels, predicts, average="weighted")
        
        epoch_acc.append(acc)
        epoch_ps.append(ps)
        epoch_rs.append(rs)
        epoch_f1.append(f1)
        trues.append(labels)
        prediction.append(predicts)

        # Backpropagation
        loss.backward()
        optimizer.step()

    trues = np.concatenate(trues)
    prediction = np.concatenate(prediction)
    accuracy = accuracy_score(trues, prediction)
    precision = precision_score(trues, prediction, average="weighted")
    recall = recall_score(trues, prediction, average="weighted")
    f1score = f1_score(trues, prediction, average="weighted")
    accuracy = accuracy * 100
    precision = precision * 100
    recall = recall * 100
    f1score = f1score * 100

    # Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc) * 100
    epoch_ps = np.mean(epoch_ps) * 100
    epoch_rs = np.mean(epoch_rs) * 100
    epoch_f1 = np.mean(epoch_f1) * 100

    wandb.log({"train_loss": epoch_loss, "train_f1": f1score})

    return epoch_loss, f1score, total_time


def val_one_epoch(model, device, criterion, val_data_loader, best_val_f1, save):
    epoch_loss = []
    epoch_acc = []
    epoch_ps = []
    epoch_rs = []
    epoch_f1 = []
    trues = []
    prediction = []
    start_time = time.time()

    model.eval()

    with torch.no_grad():
        for images, labels in val_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)

            # Calculating Loss
            loss = criterion(preds, labels)
            epoch_loss.append(loss.item())

            # Calculating Metrics
            _, predicts = preds.max(1)
            predicts = predicts.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            acc = accuracy_score(labels, predicts)
            ps = precision_score(labels, predicts, average="weighted")
            rs = recall_score(labels, predicts, average="weighted")
            f1 = f1_score(labels, predicts, average="weighted")
            epoch_acc.append(acc)
            epoch_ps.append(ps)
            epoch_rs.append(rs)
            epoch_f1.append(f1)
            trues.append(labels)
            prediction.append(predicts)
    
    trues = np.concatenate(trues)
    prediction = np.concatenate(prediction)
    accuracy = accuracy_score(trues, prediction)
    precision = precision_score(trues, prediction, average="weighted")
    recall = recall_score(trues, prediction, average="weighted")
    f1score = f1_score(trues, prediction, average="weighted")
    accuracy = accuracy * 100
    precision = precision * 100
    recall = recall * 100
    f1score = f1score * 100
    cm = confusion_matrix(trues, prediction)
    cr = classification_report(trues, prediction)

    # Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc) * 100
    epoch_ps = np.mean(epoch_ps) * 100
    epoch_rs = np.mean(epoch_rs) * 100
    epoch_f1 = np.mean(epoch_f1) * 100

    wandb.log({"val_loss": epoch_loss, "val_f1": f1score})

    # Saving best model
    if f1score > best_val_f1:
        best_val_f1 = f1score
        torch.save(model.state_dict(), f"{save}.pth")

    return epoch_loss, accuracy, precision, recall, f1score, total_time, best_val_f1, cm, cr


def dev_one_epoch(model, device, criterion, dev_data_loader):
    epoch_loss = []
    epoch_acc = []

    model.eval()

    with torch.no_grad():
        for images, labels in dev_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)  # Forward

            # Calculating Loss
            loss = criterion(preds, labels)
            epoch_loss.append(loss.item())

            # Calculating Accuracy
            _, predicts = preds.max(1)
            predicts = predicts.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            acc = accuracy_score(labels, predicts)
            epoch_acc.append(acc)

    # Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    return epoch_acc