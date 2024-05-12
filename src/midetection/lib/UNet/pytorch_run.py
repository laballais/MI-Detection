from __future__ import print_function, division
import os
import numpy as np
import glob
import torch.utils.data
import torch
import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
import shutil
import random
import time
import pandas as pd

from PIL import Image
from scipy import ndimage
from midetection.lib.UNet.Models import NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, AttU_Net_LalaSan
from midetection.lib.UNet.losses import calc_loss, threshold_predictions_v
from midetection.lib.UNet.ploting import LayerActivations
from midetection.lib.UNet.Metrics import dice_coeff
from midetection.lib.UNet.modified_sampler import ModifiedSubsetRandomSampler
from midetection.lib.UNet.Data_Loader import Images_Dataset_folder


logger_path = './logger'
initial_lr = 0.001


def get_loss_log(tr_loss, test_loss, epoch):
    return {
        'tr_loss': tr_loss,
        'test_loss': test_loss,
        'epoch': epoch
    }

def append_losses(arr, log):
    arr.append(log)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def save_history(data, dir_path, file_name):
    df = pd.DataFrame(data)
    file_path = os.join(dir_path, file_name)
    create_dir(dir_path)
    df.to_csv(file_path)
    print('Saved training details to {}'.format(file_path))


def log_text(message):
    logger = open(logger_path + "/logs.txt", "a+")
    logger.write(message)
    logger.close()


def create_directory(path, directory_name):
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)

    try:
        os.mkdir(path)
    except OSError:
        if path == logger_path:
            print('Creation of the ' + str(directory_name) +
                  ' ' + str(path) + 'failed')
        else:
            log_text('Creation of the ' + str(directory_name) +
                     ' ' + str(path) + 'failed\n')
    else:
        if path == logger_path:
            print('Successfully created the ' +
                  str(directory_name) + ' ' + str(path))
        else:
            log_text('Successfully created the ' +
                     str(directory_name) + ' ' + str(path) + '\n')


def show_plots(iteration, trainloss, iteration2, validloss):
    plt.plot(iteration, trainloss, label="Training Loss")
    plt.plot(iteration2, validloss, label="Validating Loss")
    # plt.ylim(0, 1)
    plt.xlim(0, 26)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()
    plt.show()


def run_model(batch_size=4, epoch=15, valid_size=0.15, base_paths=[]):
    #######################################################
    # Checking if logger folder exists
    #######################################################

    create_directory(logger_path, "logger directory")

    #######################################################
    # Checking if GPU is used
    #######################################################

    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        log_text('CUDA is not available. Training on CPU\n')
    else:
        log_text('CUDA is available. Training on GPU\n')

    device = torch.device("cuda:0" if train_on_gpu else "cpu")
    print("selected device is ", device)
    #######################################################
    # Setting the basic paramters of the model
    #######################################################

    print("Setting model parameters")

    log_text('batch_size = ' + str(batch_size) + '\n')
    log_text('epoch = ' + str(epoch) + '\n')

    # random_seed = random.randint(1, 100)
    random_seed = 90
    log_text('random_seed = ' + str(random_seed) + '\n')

    shuffle = True
    valid_loss_min = np.Inf
    num_workers = 4  
    lossT = []
    lossL = []
    lossL.append(np.inf)
    lossT.append(np.inf)
    epoch_valid = epoch-2
    n_iter = 1
    i_valid = 0
    train = []
    valid = []
    tr_index = []
    val_index = []

    pin_memory = False
    if train_on_gpu:
        pin_memory = True

    #######################################################
    # Setting up the model
    #######################################################

    model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, AttU_Net_LalaSan]

    def model_unet(model_input, in_channel=3, out_channel=1):
        model_test = model_input(in_channel, out_channel)
        return model_test

    # passing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary

    # change index of model_Input to select the unet type: 2 - attention
    model_test = model_unet(model_Inputs[2], 3, 1)

    model_test.to(device)

    #######################################################
    # Getting the Summary of Model
    #######################################################

    # torchsummary.summary(model_test, input_size=(3, 128, 128)) # torchsummary doesn't work for AttU_Net and R2AttU_Net : comment out
    #torchsummary.summary(model_test, input_size=(3, 224, 224))

    #######################################################
    # Passing the Dataset of Images and Labels
    #######################################################

    t_data = base_paths[0]
    l_data = base_paths[1]
    test_folderP = base_paths[2] + "*"
    test_folderL = base_paths[3] + "*"
    validating_folder = base_paths[4]
    validating_label_folder = base_paths[5]
    test_image = base_paths[6]
    test_label = base_paths[7]

    Training_Data = Images_Dataset_folder(t_data, l_data)
    Validating_Data = Images_Dataset_folder(validating_folder, validating_label_folder)

    train_dir = os.listdir(t_data)
    valid_dir = os.listdir(validating_folder)
    test_dir = os.listdir(base_paths[2])

    #######################################################
    # Giving a transformation for input data
    #######################################################

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    #######################################################
    # Giving a transformation for output data
    #######################################################

    label_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    #######################################################
    # Training Validation Split
    #######################################################

    num_train = len(Training_Data)
    train_indices = list(range(num_train))
    num_valid = len(Validating_Data)
    valid_indices = list(range(num_valid))

    # if shuffle:
    #     np.random.seed(random_seed)
    #     np.random.shuffle(train_indices)

    train_sampler = ModifiedSubsetRandomSampler(train_indices)
    valid_sampler = ModifiedSubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory,)

    valid_loader = torch.utils.data.DataLoader(Validating_Data, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=num_workers, pin_memory=pin_memory,)

    #######################################################
    # Using Adam as Optimizer
    #######################################################

    # try SGD    # , betas=(0.8,0.999)
    opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr)
    # opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)

    MAX_STEP = int(1e10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)
    # delay = 1e-6

    #######################################################
    # Creating a Folder for every data of the program
    #######################################################

    print("Creating folders for every data")

    New_folder = './model'

    create_directory(New_folder, "main directory")

    #######################################################
    # Setting the folder of saving the predictions
    #######################################################

    read_pred = './model/training_masks'
    validating_pred = './model/validating_masks'

    create_directory(read_pred, "prediction directory")
    create_directory(validating_pred, "prediction directory")

    #######################################################
    # Setting the folder of saving the pred visualization
    #######################################################

    read_predvis = './model/pred_visualization'

    create_directory(read_predvis, "visualization directory")

    #######################################################
    # checking if the model exists and if true then delete
    #######################################################

    read_model_path = './model/Unet_D_' + str(epoch) + '_' + str(batch_size)

    create_directory(read_model_path, "model directory")

    #######################################################
    # Training loop
    #######################################################

    log_text("\n")
    log_text("Training the model...\n")
    log_text("          Training '%i' images in an iteration.\n" % batch_size)
    log_text("          '%i' iterations will be done in each epoch.\n" %
             len(train_loader))

    print("Training ongoing...")

    trainIndex_sampler = {}
    validIndex_sampler = {}
    train_pred = {}
    valid_pred = {}
    best_epoch = 0
    lr = initial_lr

    for i in range(epoch):

        log_text("Training on epoch number: '%i'\n" % (i+1))

        train_loss = 0.0
        valid_loss = 0.0
        since = time.time()

        #######################################################
        # Training Data
        #######################################################

        model_test.train()
        k = 1

        log_text("          Ongoing...")
        index = 0
        train_pred[i] = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()

            y_pred = model_test(x)
            lossT = calc_loss(y_pred, y)     # Dice_loss Used

            train_loss += lossT.item() * x.size(0)

            # L2 regularization
            # l2_lambda = 0.0001
            # l2_norm = sum(p.pow(2.0).sum()
            #                 for p in model_test.parameters())
            # lossT = lossT + l2_lambda * l2_norm

            lossT.backward()
            opt.step()

            x_size = lossT.item() * x.size(0)
            k = 2   

            trainIndex_sampler[i] = train_loader.sampler.samplerIndex
            y_pred = torch.sigmoid(y_pred.cpu())
            y_pred = y_pred.detach().numpy()
            for cnt in range(batch_size):
                if (index < len(trainIndex_sampler[i])):
                    train_pred[i].append(y_pred[cnt][0])
                    index += 1

            log_text("...")
        log_text("...\n")

        scheduler.step()
        lr = scheduler.get_last_lr()

        #######################################################
        # Validation Step
        #######################################################

        log_text("          Validating...")

        model_test.eval()
        torch.no_grad()  # to increase the validation process uses less memory

        index_val = 0
        valid_pred[i] = []

        for x1, y1 in valid_loader:
            x1, y1 = x1.to(device), y1.to(device)

            #print("          Image sequence for validation: " + str(valid_loader.sampler.samplerIndex))

            y_pred1 = model_test(x1)
            lossL = calc_loss(y_pred1, y1)     # Dice_loss Used

            valid_loss += lossL.item() * x1.size(0)
            x_size1 = lossL.item() * x1.size(0)

            validIndex_sampler[i] = valid_loader.sampler.samplerIndex
            y_pred1 = torch.sigmoid(y_pred1.cpu())
            y_pred1 = y_pred1.detach().numpy()
            for cnt in range(batch_size):
                if (index_val < len(validIndex_sampler[i])):
                    valid_pred[i].append(y_pred1[cnt][0])
                    index_val += 1

            log_text("...")
        log_text("...\n")

        #######################################################
        # Saving the predictions
        #######################################################

        im_tb = Image.open(test_image)
        im_label = Image.open(test_label)
        s_tb = data_transform(im_tb)
        s_label = label_transform(im_label)
        s_label = s_label.detach().numpy()

        pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
        pred_tb = torch.sigmoid(pred_tb)
        pred_tb = pred_tb.detach().numpy()

        x1 = plt.imsave('./model/pred_visualization/img_iteration_' +
                        str(n_iter) + '_epoch_' + str(i+1) + '.png', pred_tb[0][0])

        #######################################################
        # Update training and validating losses
        #######################################################

        train_loss = train_loss / len(train_indices)
        valid_loss = valid_loss / len(valid_indices)
        train.append(train_loss)
        valid.append(valid_loss)
        tr_index.append(i+1-0.5)
        val_index.append(i+1)

        if (i+1) % 1 == 0:
            log_text('          Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n'.format(
                i + 1, epoch, train_loss, valid_loss))
            # print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n'.format(i + 1, epoch, train_loss, valid_loss))

        #######################################################
        # Early Stopping
        #######################################################
        if valid_loss <= valid_loss_min and epoch_valid >= i:  # and i_valid <= 2:
            try:
                best_epoch = i
                log_text('          Validation loss decreased ({:.6f} --> {:.6f}).  Saving model \n'.format(
                    valid_loss_min, valid_loss))
                # print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model \n'.format(valid_loss_min, valid_loss))

                torch.save(model_test.state_dict(), './model/Unet_D_' + str(epoch) + '_' + str(batch_size) +
                           '/Unet_epoch_' + str(best_epoch) + '_batchsize_' + str(batch_size) + '.pth')
                log_text('          Updating saved model epoch to ' +
                         str(best_epoch+1) + '\n')

                if round(valid_loss, 4) == round(valid_loss_min, 4):
                    print(i_valid)
                    i_valid = i_valid+1
                valid_loss_min = valid_loss
                # if i_valid ==3:
                #   break
            except:
                print('Execption found at epoch: {}'.format(i))

        #####################################
        # for kernals
        #####################################
        x1 = torch.nn.ModuleList(model_test.children())

        #####################################
        # for images
        #####################################
        x2 = len(x1)
        dr = LayerActivations(x1[x2-1])  # Getting the last Conv Layer

        img = Image.open(test_image)
        s_tb = data_transform(img)

        pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
        pred_tb = torch.sigmoid(pred_tb)
        pred_tb = pred_tb.detach().numpy()

        time_elapsed = time.time() - since
        log_text('{:.0f}m {:.0f}s\n'.format(
            time_elapsed // 60, time_elapsed % 60))
        n_iter += 1

        im_tb.close()       # added: fix memory leak
        im_label.close()    # added: fix memory leak
        img.close()         # added: fix memory leak

    #######################################################
    # Save predictions from training and val in best epoch and show plot
    #######################################################

    log_text('Saving predictions during training and validation.\n')

    # to access predictions across batches change the row index of pred[x][0]
    for cnt in range(len(train_sampler)):
        plt.imsave('./model/training_masks/' +
                   str(train_dir[trainIndex_sampler[best_epoch][cnt]]), train_pred[best_epoch][cnt])

    for cnt in range(len(valid_sampler)):
        plt.imsave('./model/validating_masks/' +
                   str(valid_dir[validIndex_sampler[best_epoch][cnt]]), valid_pred[best_epoch][cnt])

    show_plots(tr_index, train, val_index, valid)


    #######################################################
    # checking if cuda is available
    #######################################################

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    #######################################################
    # Loading the model
    #######################################################

    model_test.load_state_dict(torch.load('./model/Unet_D_' +
                                          str(epoch) + '_' + str(batch_size) +
                                          '/Unet_epoch_' + str(best_epoch)
                                          + '_batchsize_' + str(batch_size) + '.pth'))

    model_test.eval()

    #######################################################
    # opening the test folder and creating a folder for generated images
    #######################################################

    read_test_folder = glob.glob(test_folderP)
    x_sort_test = natsort.natsorted(read_test_folder)  # To sort

    read_test_folder112 = './model/gen_images'

    create_directory(read_test_folder112, "testing directory")

    # For Prediction Threshold

    read_test_folder_P_Thres = './model/testing_masks'

    create_directory(read_test_folder_P_Thres, "testing directory")

    # For Label Threshold

    read_test_folder_L_Thres = './model/label_threshold'

    create_directory(read_test_folder_L_Thres, "testing directory")

    #######################################################
    # saving the images in the test folder files
    #######################################################
    log_text("\n")
    log_text("Testing the model...")

    print("Testing the model...")

    img_test_no = 0
    test_size = len(read_test_folder)

    for i in range(test_size):
        log_text("...")
        im = Image.open(x_sort_test[i])

        im1 = im
        im_n = np.array(im1)
        im_n_flat = im_n.reshape(-1, 1)

        for j in range(im_n_flat.shape[0]):
            if im_n_flat[j] != 0:
                im_n_flat[j] = 255

        s = data_transform(im)
        # pred = model_test(s.unsqueeze(0).cuda()).cpu()     # replaced this with the line below since no cuda
        pred = model_test(s.unsqueeze(0).to(device)).cpu()
        pred = torch.sigmoid(pred)
        pred = pred.detach().numpy()

    #    pred = threshold_predictions_p(pred) #Value kept 0.01 as max is 1 and noise is very small.

        if i % test_size == 0:  # why 24?
            img_test_no = img_test_no + 1

        #img_test_no = (img_test_no % test_folder_size) + 1
        x1 = plt.imsave('./model/gen_images/' + str(test_dir[i]), pred[0][0])
    log_text("...\n")

    ####################################################
    # Calculating the Dice Score
    ####################################################

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
    ])

    read_test_folderP = glob.glob('./model/gen_images/*')
    x_sort_testP = natsort.natsorted(read_test_folderP)

    read_test_folderL = glob.glob(test_folderL)
    x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort

    dice_score123 = 0.0
    x_count = 0
    x_dice = 0
    thr = 20

    for i in range(len(read_test_folderP)):

        x = Image.open(x_sort_testP[i])
        s = data_transform(x)
        s = np.array(s)
        s = threshold_predictions_v(s)

        # zlabeled, Nlabels = ndimage.measurements.label(s)
        # label_size = [(zlabeled == label).sum() for label in range(Nlabels + 1)]
        # for label,size in enumerate(label_size):
        #     if size <= thr:
        #         s[zlabeled == label] = 0

        # save the images
        x1 = plt.imsave('./model/testing_masks/' + str(test_dir[i]), s)

        y = Image.open(x_sort_testL[i])
        s2 = data_transform(y)
        s3 = np.array(s2)
    #   s2 =threshold_predictions_v(s2)

        # save the Images
        y1 = plt.imsave('./model/label_threshold/' + str(test_dir[i]), s3)

        total = dice_coeff(s, s3)
        log_text(str(total) + '\n')

        if total <= 0.3:
            x_count += 1
        if total > 0.3:
            x_dice = x_dice + total
        dice_score123 = dice_score123 + total

    log_text('Dice Score : ' + str(dice_score123/len(read_test_folderP)) + '\n')
    print('Dice Score : ' + str(dice_score123/len(read_test_folderP)))

def get_masks():
    model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet]

    def model_unet(model_input, in_channel=3, out_channel=1):
        model_test = model_input(in_channel, out_channel)
        return model_test

    # change index of model_Input to select the unet type: 2 - attention
    model_test = model_unet(model_Inputs[0], 3, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_test.to(device)

    model_test.load_state_dict(torch.load(
        './model/Unet_D_25_4/Unet_epoch_25_batchsize_4.pth'))
    model_test.eval()

    read_test_folderP = glob.glob('./model/add_gen_images/*')
    x_sort_testP = natsort.natsorted(read_test_folderP)

    test_folderP = 'C:/Users/Lalaine/Documents/Thesis_Code/MI echo dataset/HMC-QU Echos/Frames/Final/additional input/*'
    read_test_folder = glob.glob(test_folderP)
    x_sort_test = natsort.natsorted(read_test_folder)  # To sort

    read_test_folder112 = './model/add_gen_images'

    create_directory(read_test_folder112, "testing directory")

    read_test_folderF = './model/additional'

    create_directory(read_test_folderF, "testing directory")

    print("Generating masks...")

    img_test_no = 0
    test_size = len(read_test_folder)

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    for i in range(test_size):
        im = Image.open(x_sort_test[i])

        im1 = im
        im_n = np.array(im1)
        im_n_flat = im_n.reshape(-1, 1)

        for j in range(im_n_flat.shape[0]):
            if im_n_flat[j] != 0:
                im_n_flat[j] = 255

        s = data_transform(im)
        pred = model_test(s.unsqueeze(0).to(device)).cpu()
        pred = torch.sigmoid(pred)
        pred = pred.detach().numpy()

        if i % test_size == 0:  # why 24?
            img_test_no = img_test_no + 1

        plt.imsave('./model/add_gen_images/' +
                   x_sort_test[i].split('\\', 1)[1], pred[0][0])

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
    ])

    for i in range(len(read_test_folderP)):
        x = Image.open(x_sort_testP[i])
        s = data_transform(x)
        s = np.array(s)
        s = threshold_predictions_v(s)

        zlabeled, Nlabels = ndimage.measurements.label(s)
        label_size = [(zlabeled == label).sum()
                      for label in range(Nlabels + 1)]
        for label, size in enumerate(label_size):
            if size <= 70:
                s[zlabeled == label] = 0

        # save the images
        plt.imsave('./model/additional/' +
                   x_sort_testP[i].split('\\', 1)[1], s)


def cleanup_Predictions(pred_path='./model/training_masks/', thr=20):
    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
    ])
    read_pred_folderDir = os.listdir(pred_path)
    read_pred_folder = glob.glob(pred_path + '*')
    x_sorted_pred = natsort.natsorted(read_pred_folder)

    for i in range(len(read_pred_folder)):

        x = Image.open(x_sorted_pred[i])
        s = data_transform(x)
        s = np.array(s)
        s = threshold_predictions_v(s)

        zlabeled, Nlabels = ndimage.measurements.label(s)
        label_size = [(zlabeled == label).sum()
                      for label in range(Nlabels + 1)]
        for label, size in enumerate(label_size):
            if size <= thr:
                s[zlabeled == label] = 0

        # save the images
        plt.imsave(pred_path + str(read_pred_folderDir[i]), s)

#Hello git tracking