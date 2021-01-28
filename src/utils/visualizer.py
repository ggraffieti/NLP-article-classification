import matplotlib.pyplot as plt
from utils.training_utils import load_metrics
import os


def visualize_losses(loss_folder, device):
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(loss_folder + '/metrics.pt', device)
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(loss_folder, "losses.png"))
