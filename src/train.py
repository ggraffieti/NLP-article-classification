import torch
import torch.optim as optim
from models.bert import Bert
from data.loaders import get_train_valid_test_new
from utils.training_utils import save_metrics, save_checkpoint, load_checkpoint
from utils.visualizer import visualize_losses
from test import evaluate
import time


def train(model, optimizer, train_loader, valid_loader, num_epochs, eval_every,
          file_path, best_valid_loss=float("Inf")):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for labels, title, text, titletext in train_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            titletext = titletext.type(torch.LongTensor)
            titletext = titletext.to(device)
            output = model(titletext, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for val_labels, val_title, val_text, val_titletext in valid_loader:
                        val_labels = val_labels.type(torch.LongTensor)
                        val_labels = val_labels.to(device)
                        val_titletext = val_titletext.type(torch.LongTensor)
                        val_titletext = val_titletext.to(device)
                        output = model(val_titletext, val_labels)
                        loss, _ = output

                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(file_path + '/' + 'metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    bert_model = "bert-base-uncased"
    bert = Bert(bert_model=bert_model).to(device)

    train_iter, valid_iter, test_iter = get_train_valid_test_new(bert_model=bert_model, max_seq_lenght=128)

    opt = optim.Adam(bert.parameters(), lr=2e-5)
    init_time = time.time()

    train(model=bert, optimizer=opt, train_loader=train_iter, valid_loader=valid_iter, num_epochs=5,
          eval_every=len(train_iter) // 2, file_path="../data/models/")

    tot_time = time.time() - init_time
    print("time taken:", int(tot_time // 60), "minutes", int(tot_time % 60), "seconds")

    # visualize_losses("../data/models", device)

    best_model = Bert(bert_model=bert_model).to(device)

    load_checkpoint("../data/models" + '/model.pt', best_model, device)

    evaluate(best_model, test_iter, device)
