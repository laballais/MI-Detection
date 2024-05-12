# import the necessary libraries
import time
from sklearn.metrics import precision_score
import torch
import torch.nn as nn
import torchvision.utils
from midetection.lib.siamese_net.siamese_net import config
from midetection.lib.siamese_net.siamese_net.utils import imshow, show_plot, show_plots
from midetection.lib.siamese_net.siamese_net.metrics import accuracy_score, sensitivity_score, specificity_score, precision_score, plot_confusionMatrix
import torchvision
import os
from midetection.Utils.utilities import log_text
from midetection.lib.siamese_net.siamese_net.model_bce import SiameseNetwork
# from midetection.lib.siamese_net.siamese_net.dataset import train_dataloader, test_dataloader, validating_dataloader
from midetection.lib.siamese_net.siamese_net.dataset import get_data_loader
import random
from midetection.lib.siamese_net.siamese_net.contrastive import ContrastiveLoss
import torch.nn.functional as F
import numpy as np


if os.path.isfile("./logger/prediction directory.txt"):
    os.remove("./logger/prediction directory.txt")


train_on_gpu = torch.cuda.is_available()                                                                                                                   
device = torch.device("cuda:0" if train_on_gpu else "cpu")
i_valid = 0
valid_loss_min = np.Inf

SEED = random.randint(1, 200)
# SEED = 51
# SEED = 18
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

log_text("Random seed: " + str(SEED) + "\n", "prediction directory.txt")
log_text("Epoch: " + str(config.epochs) + "\n", "prediction directory.txt")
log_text("Batch Size: " + str(config.batch_size) + "\n", "prediction directory.txt")

net = SiameseNetwork().to(device)
# Decalre Loss Function
criterion = nn.BCEWithLogitsLoss()
# criterion = ContrastiveLoss()
# Declare Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

# implement early stopping
def stop_early(model, valid_loss, i):
    epoch_valid = config.epochs-2
    global valid_loss_min
    global i_valid
    if valid_loss <= valid_loss_min and epoch_valid >= i: # and i_valid <= 2:
        log_text('      Validation loss decreased ({:.6f} --> {:.6f}).  Saving model \n'.format(valid_loss_min, valid_loss), "prediction directory.txt")

        torch.save(model.state_dict(), "model.pth")
        log_text("Model saved.\n", "prediction directory.txt")
        best_epoch = i
        log_text('      Updating saved model epoch to ' + str(best_epoch) + '\n', "prediction directory.txt")

        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid = i_valid+1
        valid_loss_min = valid_loss

# validate the model
def validate_model(wallMotion, myocardialThickening, validating_dataloader):
    log_text("Validating the model...\n", "prediction directory.txt")
    recall = []
    prec = []
    net.eval()
    torch.no_grad() #to increase the validation process uses less memory
    prediction = []
    groundtruth = []
    valid_loss = 0
    for i, data in enumerate(validating_dataloader, 0):
        img0, img1, wm, mt, label, im_index = data
        img0, img1, label, wm, mt = img0.to(device), img1.to(device), label.to(device), wm.to(device), mt.to(device)
        optimizer.zero_grad()
        pred = net(img0, img1, wm, mt)

        # print(str(out1))
        bce_loss = criterion(pred, label)

        valid_loss += bce_loss.item()

        pred = torch.sigmoid(pred)
        label = label
        for i, result in enumerate(pred):
            if (result > 0.5):    # dissimilar: MI
                prediction.append(1)
            else:
                prediction.append(0)

            if label[i] == torch.FloatTensor([[0]]).to(device):
                groundtruth.append(0)
            else:
                groundtruth.append(1)
    
    sensitivity = sensitivity_score(prediction, groundtruth)
    recall.append(sensitivity/100)
    precision = precision_score(prediction, groundtruth)
    prec.append(precision/100)

    log_text("      Current loss {}\n".format(valid_loss / len(validating_dataloader)), "prediction directory.txt")
    log_text("      Current precision {}\n".format(precision), "prediction directory.txt")
    log_text("      Current sensitivity {}\n".format(sensitivity), "prediction directory.txt")
    log_text("      Current accuracy {}\n".format(accuracy_score(prediction, groundtruth)), "prediction directory.txt")
    log_text("      Current specificity {}\n".format(specificity_score(prediction, groundtruth)), "prediction directory.txt")

    return valid_loss / len(validating_dataloader)

# train the model
def run_model(wallMotion, myocardialThickening, data):
    print("Training the model...")
    log_text("Training the model...\n", "prediction directory.txt")
    recall = []
    prec = []
    loss = []
    validLoss = []
    counter = []
    train_dataloader, _, validating_dataloader = get_data_loader(wallMotion, myocardialThickening, data)
    net.train()
    for epoch in range(1, config.epochs+1):
        log_text("Epoch num: " + str(epoch) + "\n", "prediction directory.txt")
        prediction = []
        groundtruth = []
        epoch_loss = 0
        since = time.time()
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, wm, mt, label, im_index = data
            img0, img1, label, wm, mt = img0.to(device), img1.to(device), label.to(device), wm.to(device), mt.to(device)
            optimizer.zero_grad()
            pred = net(img0, img1, wm, mt)
            bce_loss = criterion(pred, label)
            bce_loss.backward()
            optimizer.step()

            epoch_loss += bce_loss.item()

            pred = torch.sigmoid(pred)
            label = label
            for i, result in enumerate(pred):
                if (result > 0.5):    # dissimilar: MI
                    prediction.append(1)
                else:
                    prediction.append(0)

                if label[i] == torch.FloatTensor([[0]]).to(device):
                    groundtruth.append(0)
                else:
                    groundtruth.append(1)
      
        log_text("Epoch {}:\n      Current loss {}\n".format(epoch, epoch_loss / len(train_dataloader)), "prediction directory.txt")
        counter.append(epoch)
        train_loss = epoch_loss / len(train_dataloader)
        loss.append(train_loss)

        sensitivity = sensitivity_score(prediction, groundtruth)
        recall.append(sensitivity/100)
        precision = precision_score(prediction, groundtruth)
        prec.append(precision/100)

        log_text("      Current precision {}\n".format(precision), "prediction directory.txt")
        log_text("      Current sensitivity {}\n".format(sensitivity), "prediction directory.txt")
        log_text("      Current accuracy {}\n".format(accuracy_score(prediction, groundtruth)), "prediction directory.txt")
        log_text("      Current specificity {}\n".format(specificity_score(prediction, groundtruth)), "prediction directory.txt")

        valid_loss = validate_model(wallMotion, myocardialThickening, validating_dataloader)
        validLoss.append(valid_loss)
        stop_early(net, valid_loss, epoch)

        time_elapsed = time.time() - since        
        log_text('{:.0f}m {:.2f}s\n'.format(time_elapsed // 60, time_elapsed % 60), "prediction directory.txt")
    # show_plots(counter, loss, validLoss)
    # print(str(TPR))
    # print(str(FPR))
    # show_plot(recall, prec)
    return net, recall, prec, loss, validLoss, counter
    

def train_model(wallMotion, myocardialThickening, data):
    start = time.time()   
    model, recall, prec, loss, validLoss, counter = run_model(wallMotion, myocardialThickening, data)

    # torch.save(model.state_dict(), "model.pth")
    # log_text("Model saved.\n", "prediction directory.txt")
    # print("Model Saved Successfully.")

    duration = time.time() - start
    log_text('Training duration: {:.0f}m {:.2f}s\n'.format(duration // 60, duration % 60), "prediction directory.txt")
    return model, recall, prec, loss, validLoss, counter


def test_model(wallMotion, myocardialThickening, data, net):
    if(not net):
        net.load_state_dict(torch.load('./model.pth'))
    net.eval()
    print("Testing the model...")
    log_text("\nTesting the model...\n", "prediction directory.txt")
    _, test_dataloader, _ = get_data_loader(wallMotion, myocardialThickening, data)
    prediction = []
    groundtruth = []
    # print("TEST length: ", len(test_dataloader))
    for i, data in enumerate(test_dataloader, 0):
        log_text(str(i+1) + "\n", "prediction directory.txt")
        x0, x1, wm, mt, ll, im_index = data
        predict = net(x0.to(device), x1.to(device), wm.to(device), mt.to(device))

        predict = torch.sigmoid(predict.cpu())
        ll = ll.cpu()
        for j, result in enumerate(predict):
            if (result > 0.5):    # dissimilar: MI
                pred = "Myocardial Infarction"
                prediction.append(1)
            else:
                pred = "Non Myocardial Infarction"
                prediction.append(0)

            if ll[j] == torch.FloatTensor([[1]]):
                label = "Myocardial Infarction"         # label = "Original Pair Of Signature"
                groundtruth.append(1)
            else:
                label = "Non Myocardial Infarction"     # label = "Forged Pair Of Signature"
                groundtruth.append(0)
    
            # print("Case No: " + str(im_index[j]))
            # print("Predicted Label:-" + str(pred))
            # print("Actual Label:-" + str(label))
            # log_text("  Case No: " + str(im_index[j]) + "\n", "prediction directory.txt")
            # log_text("     Predicted Label: " + pred + "\n", "prediction directory.txt")
            # log_text("     Actual Label: " + label + "\n", "prediction directory.txt")

    accuracy = accuracy_score(prediction, groundtruth)
    sensitivity = sensitivity_score(prediction, groundtruth)
    precision = precision_score(prediction, groundtruth)
    specificity = specificity_score(prediction, groundtruth)
    plot_confusionMatrix(prediction, groundtruth)

    print("")
    print("Accuracy: " + str(accuracy) + " %")
    print("Sensitivity: " + str(sensitivity) + " %")
    print("Precision: " + str(precision) + " %")
    print("Specificity: " + str(specificity) + " %")

    log_text("\n", "prediction directory.txt")
    log_text("Accuracy: " + str(accuracy) + "\n", "prediction directory.txt")
    log_text("Sensitivity: " + str(sensitivity) + "\n", "prediction directory.txt")
    log_text("Precision: " + str(precision) + "\n", "prediction directory.txt")
    log_text("Specificity: " + str(specificity) + "\n", "prediction directory.txt")
    return accuracy, sensitivity, precision, specificity, prediction, groundtruth