import argparse
import warnings
from dataset import get_data_loaders
from model import get_model_optimizer_scheduler
from train import train_one_epoch, dev_one_epoch, val_one_epoch
from utils import setup_device, set_seed, setup_wandb, print_final_metrics, save_metrics_to_csv_and_wandb, print_model_info

def args_parser():
    parser = argparse.ArgumentParser(description='Your script description')
    # data parameters
    # path on twcc:/work/u9562361/crisis_vision_benchmarks/damage_severity
    parser.add_argument('--data_dir', type=str, default='C:/crisis_vision_benchmarks/data/damage_severity', help='path to data directory')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--image_res', type=int, default=224, help='resolution of images')
    # training parameter
    parser.add_argument('--epochs', type=int, default=150, help='number of training epochs')
    parser.add_argument('--lr', type=int, default=1e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=int, default=1e-3, help='weight decay')
    parser.add_argument('--seed', type=int, default=7, help='seed for reproducibility')
    parser.add_argument('--save', type=str, default='test', help='file save')
    # Model selection
    parser.add_argument('--model', default='effnet', help='choose the model') # effnet
    parser.add_argument('--freeze', type=bool, default=False, help='Freeze backbone parameters')
    return parser.parse_args()

def main():
    args = args_parser()

    # Setup
    device = setup_device()
    set_seed(args.seed)
    warnings.filterwarnings("ignore")
    setup_wandb(args) 

    # Data Loading
    trainloader, testloader, devloader, class_names = get_data_loaders(args)

    # Model, Criterion and Optimizer
    model, criterion, optimizer, lr_scheduler = get_model_optimizer_scheduler(args, class_names, device) 
    print_model_info(model)


    # Training loop
    print('checkpoint: start training...')
    output_list = []
    best_val_acc, best_val_ps, best_val_rs, best_val_f1 = 0, 0, 0, 0

    for epoch in range(args.epochs):
        train_loss, train_f1, train_time = train_one_epoch(model, device, criterion, optimizer, trainloader)
        dev_acc = dev_one_epoch(model, device, criterion, devloader)
        val_loss, val_acc, val_ps, val_rs, val_f1, val_time, best_val_f1, cm, cr = val_one_epoch(model, device, criterion, testloader, 
                                                                                                 best_val_f1, args.save)
        lr_scheduler.step(dev_acc)
        best_val_acc = max(val_acc, best_val_acc)
        best_val_ps = max(val_ps, best_val_ps)
        best_val_rs = max(val_rs, best_val_rs)

        total_time = train_time + val_time
        output_str = f"Epoch {epoch+1}/{args.epochs} - loss: {train_loss:.4f} - f1-score: {train_f1:.2f}% - val_loss: {val_loss:.4f} - val_f1-score: {val_f1:.2f}% - time: {total_time:.2f}s"
        output_list.append(output_str)
        print(output_str)
    
    save_metrics_to_csv_and_wandb(output_list, args.save + '.csv', "output_log")

    print_final_metrics(best_val_acc, best_val_ps, best_val_rs, best_val_f1, cr, cm, class_names, args)


if __name__ == "__main__":
    main()