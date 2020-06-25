import time
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from __main__ import *

def MNisttrainNN(net, batch_size, epochs, lr,train_set,train_sampler,val_sampler,test_set,test_sampler, classes):
    print("Hyper Parameters")
    print("############################################")
    print("batch size: " + str(batch_size))
    print("epochs: " + str(epochs))
    print("learning rate: " + str(lr))
    print("############################################")
    print("Thinking...")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,sampler=train_sampler, num_workers=2)

    val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)

    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss = NN.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr)
    # Time for printing
    StartTrainTime = time.time()

    costs = []
    points = []
    counter = 0
    for epoch in range(epochs):
        print_every = n_batches // 10
        totalLoss = 0
        ValidationTotalLoss = 0
        epoch_start = time.time()

        for inputs, labels in val_loader:
            inputs, labels = Variable(inputs), Variable(labels)

            ValO = net(inputs)
            LossVal = loss(ValO, labels)
            ValidationTotalLoss += LossVal.data

        for i, data in enumerate(train_loader, 0):
            runloss = 0.0
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            runloss += loss_size.data
            totalLoss += loss_size.data

            if counter % 5 == 0:
                points.append(counter)
                costs.append(runloss)
            counter +=1
        print("Epoch #" + str(epoch + 1) + " Training Time: " + str(round(time.time() - epoch_start, 2)))


    print("Total Training Time: " + str(round(time.time() - StartTrainTime,2)))

    plt.scatter(points, np.squeeze(costs))
    plt.plot(points, np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations [epochs*(20,000/batch_size)]')
    plt.title("Learning Rate = " + str(lr) + " ,Batch Size = " + str(batch_size) + " ,Epochs = " + str(epochs))
    plt.show()

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, sampler=test_sampler, num_workers=2)
    testcount = 0
    correct = 0
    top3correct = 0

    top1error_0 = 0
    top1error_1 = 0
    top1error_2 = 0
    top1error_3 = 0
    top1error_4 = 0
    top1error_5 = 0
    top1error_6 = 0
    top1error_7 = 0
    top1error_8 = 0
    top1error_9 = 0

    top3error_0 = 0
    top3error_1 = 0
    top3error_2 = 0
    top3error_3 = 0
    top3error_4 = 0
    top3error_5 = 0
    top3error_6 = 0
    top3error_7 = 0
    top3error_8 = 0
    top3error_9 = 0

    print("########################## TESTING DATA TOP 1 ##################################")
    all_preds = list([])
    all_labels = list([])
    for inputs, labels in test_loader:
        inputs, labels = Variable(inputs), Variable(labels)

        TestO = net(inputs)
        _, predicted = torch.max(TestO.data, 1)
        all_preds.append(int(predicted))
        all_labels.append(int(labels))

        top3 = torch.topk(TestO.data, 3)
        top3 = top3.indices.tolist()

        # TOP1 accuracy
        if predicted == int(labels):
            correct += 1
        testcount += 1
        # TOP3 accuracy
        if int(labels) in top3[0]:
            top3correct += 1

        # TOP1 error rate aka percentage of time it was NOT correct
        if predicted != int(labels):
            if predicted == 0 :
                top1error_1 += 1
            elif predicted == 1 :
                top1error_2 += 1
            elif predicted == 2:
                top1error_3 += 1
            elif predicted == 3:
                top1error_4 +=1
            elif predicted == 4:
                top1error_5 +=1
            elif predicted == 5:
                top1error_6 +=1
            elif predicted == 6:
                top1error_7 += 1
            elif predicted == 7:
                top1error_8 += 1
            elif predicted == 8:
                top1error_9 += 1
            elif predicted == 9:
                top1error_0 += 1

            # TOP3 error rate aka percentage of time it was NOT correct
            if 0 not in top3[0] and int(labels) == 0:
                top3error_1 += 1
            elif 1 not in top3[0] and int(labels) == 1:
                top3error_2 += 1
            elif 2 not in top3[0] and int(labels) == 2:
                top3error_3 += 1
            elif 3 not in top3[0] and int(labels) == 3:
                top3error_4 += 1
            elif 4 not in top3[0] and int(labels) == 4:
                top3error_5 += 1
            elif 5 not in top3[0] and int(labels) == 5:
                top3error_6 += 1
            elif 6 not in top3[0] and int(labels) == 6:
                top3error_7 += 1
            elif 7 not in top3[0] and int(labels) == 7:
                top3error_8 += 1
            elif 8 not in top3[0] and int(labels) == 8:
                top3error_9 += 1
            elif 9 not in top3[0] and int(labels) == 9:
                top3error_0 += 1
    all_preds = torch.FloatTensor(all_preds)
    all_labels = torch.FloatTensor(all_labels)
    stacked = torch.stack((all_labels, all_preds), dim=1)

    stacked[0].tolist()
    cmt = torch.zeros(10, 10, dtype=torch.int64)

    for p in stacked:
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1
    cmt = cmt.numpy()

    images = ["1","2","3","4","5","6","7","8","9","0"]
    df_cm = pd.DataFrame(cmt, index=[i for i in images],
                         columns=[i for i in images])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Test Data Confusion Matrix")
    plt.show()
    time.sleep(2)

    print("TOP 1 Percentage of predictions that WERE correct:  " + str(round(100 * (correct / testcount))))
    # TOP 1 ERROR RATE
    er_1 = 100 * (top1error_1 / testcount)
    er_2 = 100 * (top1error_2 / testcount)
    er_3 = 100 * (top1error_3 / testcount)
    er_4 = 100 * (top1error_4 / testcount)
    er_5 = 100 * (top1error_5 / testcount)
    er_6 = 100 * (top1error_6 / testcount)
    er_7 = 100 * (top1error_7 / testcount)
    er_8 = 100 * (top1error_8 / testcount)
    er_9 = 100 * (top1error_9 / testcount)
    er_0 = 100 * (top1error_0 / testcount)
    print("Percentage of predictions that were NOT correct, 1:  " + str(round(er_1,2)))
    print("Percentage of predictions that were NOT correct, 2:  " + str(round(er_2,2)))
    print("Percentage of predictions that were NOT correct, 3:  " + str(round(er_3,2)))
    print("Percentage of predictions that were NOT correct, 4:  " + str(round(er_4,2)))
    print("Percentage of predictions that were NOT correct, 5:  " + str(round(er_5,2)))
    print("Percentage of predictions that were NOT correct, 6:  " + str(round(er_6,2)))
    print("Percentage of predictions that were NOT correct, 7:  " + str(round(er_7,2)))
    print("Percentage of predictions that were NOT correct, 8:  " + str(round(er_8,2)))
    print("Percentage of predictions that were NOT correct, 9:  " + str(round(er_9,2)))
    print("Percentage of predictions that were NOT correct, 0:  " + str(round(er_0,2)))

    print("########################## TESTING DATA TOP 3 ##################################")
    print("TOP 3 Percentage of predictions that WERE correct:  " + str(round(100 * (top3correct / testcount),2)))

    # TOP 1 ERROR RATE
    er_1 = 100 * (top3error_1 / testcount)
    er_2 = 100 * (top3error_2 / testcount)
    er_3 = 100 * (top3error_3 / testcount)
    er_4 = 100 * (top3error_4 / testcount)
    er_5 = 100 * (top3error_5 / testcount)
    er_6 = 100 * (top3error_6 / testcount)
    er_7 = 100 * (top3error_7 / testcount)
    er_8 = 100 * (top3error_8 / testcount)
    er_9 = 100 * (top3error_9 / testcount)
    er_0 = 100 * (top3error_0 / testcount)
    print("Percentage of predictions that were NOT correct, 1:  " + str(round(er_1,2)))
    print("Percentage of predictions that were NOT correct, 2:  " + str(round(er_2,2)))
    print("Percentage of predictions that were NOT correct, 3:  " + str(round(er_3,2)))
    print("Percentage of predictions that were NOT correct, 4:  " + str(round(er_4,2)))
    print("Percentage of predictions that were NOT correct, 5:  " + str(round(er_5,2)))
    print("Percentage of predictions that were NOT correct, 6:  " + str(round(er_6,2)))
    print("Percentage of predictions that were NOT correct, 7:  " + str(round(er_7,2)))
    print("Percentage of predictions that were NOT correct, 8:  " + str(round(er_8,2)))
    print("Percentage of predictions that were NOT correct, 9:  " + str(round(er_9,2)))
    print("Percentage of predictions that were NOT correct, 0:  " + str(round(er_0,2)))

########################################### Training ################################################
    traincount = 0
    correct = 0
    top3correct = 0

    top1error_0 = 0
    top1error_1 = 0
    top1error_2 = 0
    top1error_3 = 0
    top1error_4 = 0
    top1error_5 = 0
    top1error_6 = 0
    top1error_7 = 0
    top1error_8 = 0
    top1error_9 = 0

    top3error_0 = 0
    top3error_1 = 0
    top3error_2 = 0
    top3error_3 = 0
    top3error_4 = 0
    top3error_5 = 0
    top3error_6 = 0
    top3error_7 = 0
    top3error_8 = 0
    top3error_9 = 0

    print("########################## TRAINING DATA TOP 1 ##################################")
    all_preds = list([])
    all_labels = list([])
    for inputs, labels in test_loader:
        inputs, labels = Variable(inputs), Variable(labels)

        TestO = net(inputs)
        _, predicted = torch.max(TestO.data, 1)
        all_preds.append(int(predicted))
        all_labels.append(int(labels))

        top3 = torch.topk(TestO.data, 3)
        top3 = top3.indices.tolist()

        # TOP1 accuracy
        if predicted == int(labels):
            correct += 1
        traincount += 1
        # TOP3 accuracy
        if int(labels) in top3[0]:
            top3correct += 1

        # TOP1 error rate aka percentage of time it was NOT correct
        if predicted != int(labels):
            if predicted == 0 :
                top1error_1 += 1
            elif predicted == 1 :
                top1error_2 += 1
            elif predicted == 2:
                top1error_3 += 1
            elif predicted == 3:
                top1error_4 +=1
            elif predicted == 4:
                top1error_5 +=1
            elif predicted == 5:
                top1error_6 +=1
            elif predicted == 6:
                top1error_7 += 1
            elif predicted == 7:
                top1error_8 += 1
            elif predicted == 8:
                top1error_9 += 1
            elif predicted == 9:
                top1error_0 += 1

            # TOP3 error rate aka percentage of time it was NOT correct
            if 0 not in top3[0] and int(labels) == 0:
                top3error_1 += 1
            elif 1 not in top3[0] and int(labels) == 1:
                top3error_2 += 1
            elif 2 not in top3[0] and int(labels) == 2:
                top3error_3 += 1
            elif 3 not in top3[0] and int(labels) == 3:
                top3error_4 += 1
            elif 4 not in top3[0] and int(labels) == 4:
                top3error_5 += 1
            elif 5 not in top3[0] and int(labels) == 5:
                top3error_6 += 1
            elif 6 not in top3[0] and int(labels) == 6:
                top3error_7 += 1
            elif 7 not in top3[0] and int(labels) == 7:
                top3error_8 += 1
            elif 8 not in top3[0] and int(labels) == 8:
                top3error_9 += 1
            elif 9 not in top3[0] and int(labels) == 9:
                top3error_0 += 1
    all_preds = torch.FloatTensor(all_preds)
    all_labels = torch.FloatTensor(all_labels)
    stacked = torch.stack((all_labels, all_preds), dim=1)

    stacked[0].tolist()
    cmt = torch.zeros(10, 10, dtype=torch.int64)

    for p in stacked:
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1
    cmt = cmt.numpy()

    images = ["1","2","3","4","5","6","7","8","9","0"]
    df_cm = pd.DataFrame(cmt, index=[i for i in images],
                         columns=[i for i in images])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Training Data Confusion Matrix")
    plt.show()
    time.sleep(2)

    print("TOP 1 Percentage of predictions that WERE correct:  " + str(100 * (correct / traincount)))
    # TOP 1 ERROR RATE
    er_1 = 100 * (top1error_1 / traincount)
    er_2 = 100 * (top1error_2 / traincount)
    er_3 = 100 * (top1error_3 / traincount)
    er_4 = 100 * (top1error_4 / traincount)
    er_5 = 100 * (top1error_5 / traincount)
    er_6 = 100 * (top1error_6 / traincount)
    er_7 = 100 * (top1error_7 / traincount)
    er_8 = 100 * (top1error_8 / traincount)
    er_9 = 100 * (top1error_9 / traincount)
    er_0 = 100 * (top1error_0 / traincount)
    print("Percentage of predictions that were NOT correct, 1:  " + str(round(er_1,2)))
    print("Percentage of predictions that were NOT correct, 2:  " + str(round(er_2,2)))
    print("Percentage of predictions that were NOT correct, 3:  " + str(round(er_3,2)))
    print("Percentage of predictions that were NOT correct, 4:  " + str(round(er_4,2)))
    print("Percentage of predictions that were NOT correct, 5:  " + str(round(er_5,2)))
    print("Percentage of predictions that were NOT correct, 6:  " + str(round(er_6,2)))
    print("Percentage of predictions that were NOT correct, 7:  " + str(round(er_7,2)))
    print("Percentage of predictions that were NOT correct, 8:  " + str(round(er_8,2)))
    print("Percentage of predictions that were NOT correct, 9:  " + str(round(er_9,2)))
    print("Percentage of predictions that were NOT correct, 0:  " + str(round(er_0,2)))

    print("########################## TRAINING DATA TOP 3 ##################################")
    print("TOP 3 Percentage of predictions that WERE correct:  " + str(round(100 * (top3correct / traincount),2)))

    # TOP 1 ERROR RATE
    er_1 = 100 * (top3error_1 / traincount)
    er_2 = 100 * (top3error_2 / traincount)
    er_3 = 100 * (top3error_3 / traincount)
    er_4 = 100 * (top3error_4 / traincount)
    er_5 = 100 * (top3error_5 / traincount)
    er_6 = 100 * (top3error_6 / traincount)
    er_7 = 100 * (top3error_7 / traincount)
    er_8 = 100 * (top3error_8 / traincount)
    er_9 = 100 * (top3error_9 / traincount)
    er_0 = 100 * (top3error_0 / traincount)
    print("Percentage of predictions that were NOT correct, 1:  " + str(round(er_1,2)))
    print("Percentage of predictions that were NOT correct, 2:  " + str(round(er_2,2)))
    print("Percentage of predictions that were NOT correct, 3:  " + str(round(er_3,2)))
    print("Percentage of predictions that were NOT correct, 4:  " + str(round(er_4,2)))
    print("Percentage of predictions that were NOT correct, 5:  " + str(round(er_5,2)))
    print("Percentage of predictions that were NOT correct, 6:  " + str(round(er_6,2)))
    print("Percentage of predictions that were NOT correct, 7:  " + str(round(er_7,2)))
    print("Percentage of predictions that were NOT correct, 8:  " + str(round(er_8,2)))
    print("Percentage of predictions that were NOT correct, 9:  " + str(round(er_9,2)))
    print("Percentage of predictions that were NOT correct, 0:  " + str(round(er_0,2)))
