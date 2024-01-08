import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import wandb
from sklearn.metrics import ConfusionMatrixDisplay

def setup_device():
    device = torch.device("cuda")
    parallel = torch.cuda.device_count() >= 1
    gpu_count = torch.cuda.device_count()
    print('parallel:', parallel)
    print("Using", gpu_count, "GPUs")
    return device

def save_metrics_to_csv_and_wandb(output_list, filename, wandb_log_name):
    output_list = [line.split(" - ") for line in output_list]
    output_df = pd.DataFrame(output_list, columns=["epoch", "train loss", "train f1", "val loss", "val f1", "time"])
    output_df.to_csv(filename, index=False)
    wandb.log({wandb_log_name: wandb.Table(data=output_df)})

def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

def setup_wandb(args):
    wandb.login()
    wandb.init(project='exp', config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
    }, name=args.save) # runs name same as file save

def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters in the model: {total_params}')
    wandb.log({"total_params": total_params})
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters in the model: {total_trainable_params}')
    wandb.log({"total_trainable_params": total_trainable_params})
    # print(model)

def print_final_metrics(best_val_acc, best_val_ps, best_val_rs, best_val_f1, cr, cm, class_names, args):
    print()
    print(f'Accuracy : {best_val_acc:.2f}%', f'Precision : {best_val_ps:.2f}%', f'Recall : {best_val_rs:.2f}%', f'F1-Score : {best_val_f1:.2f}%')
    print()
    print(cr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.savefig(f"{args.save}-CM.png")
    wandb.log({"confusion_matrix": [wandb.Image(f"{args.save}-CM.png")]})
    print('checkpoint: done!!!')