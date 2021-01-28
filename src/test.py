import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from models.bert import Bert
from utils.training_utils import load_checkpoint


def evaluate(model, test_loader, device):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for labels, title, text, titletext in test_loader:
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            titletext = titletext.type(torch.LongTensor)
            titletext = titletext.to(device)
            output = model(titletext, labels)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))

    # cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # ax = plt.subplot()
    # sns.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt="d")
    #
    # ax.set_title('Confusion Matrix')
    #
    # ax.set_xlabel('Predicted Labels')
    # ax.set_ylabel('True Labels')
    #
    # ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    # ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
