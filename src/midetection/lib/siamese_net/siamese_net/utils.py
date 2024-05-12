import numpy as np
import matplotlib.pyplot as plt


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(
            75,
            8,
            text,
            style="italic",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 10},
        )
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    # plt.ylabel("Precision")
    # plt.xlabel("Recall")
    plt.show()

def show_plots(iteration, trainloss, validloss):
    plt.plot(iteration, trainloss, label="Training Loss")
    # plt.plot(iteration, validloss, label="Validating Loss")
    # plt.ylim(0, 1)
    # plt.xlim(0, 1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.legend()
    plt.show()


# assumed data is a dictionary of the form {key: array_of_features}
def normalize_data1(data):
    all_min1 = 99
    all_min2 = 99
    all_min3 = 99
    all_min5 = 99
    all_min6 = 99
    all_min7 = 99
    all_max1 = 0
    all_max2 = 0
    all_max3 = 0
    all_max5 = 0
    all_max6 = 0
    all_max7 = 0

    for i, key in enumerate(data):
        data_min1 = data[key][0]
        if data_min1 < all_min1:
            all_min1 = data_min1
        data_min2 = data[key][1]
        if data_min2 < all_min2:
            all_min2 = data_min2
        data_min3 = data[key][2]
        if data_min3 < all_min3:
            all_min3 = data_min3
        data_min5 = data[key][3]
        if data_min5 < all_min5:
            all_min5 = data_min5
        data_min6 = data[key][4]
        if data_min6 < all_min6:
            all_min6 = data_min6
        data_min7 = data[key][5]
        if data_min7 < all_min7:
            all_min7 = data_min7
        
        data_max1 = data[key][0]
        if data_max1 > all_max1:
            all_max1 = data_max1
        data_max2 = data[key][1]
        if data_max2 > all_max2:
            all_max2 = data_max2
        data_max3 = data[key][2]
        if data_max3 > all_max3:
            all_max3 = data_max3
        data_max5 = data[key][3]
        if data_max5 > all_max5:
            all_max5 = data_max5
        data_max6 = data[key][4]
        if data_max6 > all_max6:
            all_max6 = data_max6
        data_max7 = data[key][5]
        if data_max7 > all_max7:
            all_max7 = data_max7
    
    for i, key in enumerate(data):
        data[key][0] = (data[key][0] - all_min1) / (all_max1 - all_min1)
        data[key][1] = (data[key][1] - all_min2) / (all_max2 - all_min2)
        data[key][2] = (data[key][2] - all_min3) / (all_max3 - all_min3)
        data[key][3] = (data[key][3] - all_min5) / (all_max5 - all_min5)
        data[key][4] = (data[key][4] - all_min6) / (all_max6 - all_min6)
        data[key][5] = (data[key][5] - all_min7) / (all_max7 - all_min7)

    return data

def normalize_data(data):
    all_min = 99
    all_max = 0

    for i, key in enumerate(data):
        data_min = min( data[key])
        if data_min < all_min:
            all_min = data_min
        
        data_max = max(data[key])
        if data_max > all_max:
            all_max = data_max
    
    for i, key in enumerate(data):
        for j in range(6):
            data[key][j] = (data[key][j] - all_min) / (all_max - all_min)

    return data

def normalize_data2(data):
    for i, key in enumerate(data):
        data_min = min(data[key])      
        data_max = max(data[key])

        for j in range(6):
            data[key][j] = (data[key][j] - data_min) / (data_max - data_min)

    return data
