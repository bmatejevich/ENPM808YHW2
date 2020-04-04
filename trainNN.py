import time
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from __main__ import *



def trainNet(net, batch_size, epochs, lr,train_set,train_sampler,val_sampler,test_set,test_sampler, classes):
    print("Hyper Parameters")
    print("############################################")
    print("batch size: ", batch_size)
    print("epochs: ", epochs)
    print("learning rate: ", lr)
    print("############################################")
    print("Thinking...")


    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,sampler=train_sampler, num_workers=2)

    val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)

    n_batches = len(train_loader)

    loss = NN.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adadelta(net.parameters(),lr)

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

        #print("Epoch #"+ str(epoch+1))



        for i, data in enumerate(train_loader, 0):

            runloss = 0.0
            inputs, labels = data

            inputs, labels = Variable(inputs), Variable(labels)


            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            #_, predicted = torch.max(outputs.data, 1)

            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            runloss = loss_size.data
            totalLoss += loss_size.data



            if counter%5 ==0:
                points.append(counter)
                costs.append(runloss)

            counter+=1
        print("Epoch #"+str(epoch+1) +" Training Time: ", round(time.time() - epoch_start,2))



    print("Total Training Time: ", round(time.time() - StartTrainTime,2))

    plt.scatter(points, np.squeeze(costs))
    plt.plot(points, np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations [epochs*(50,000/batch_size)]')
    plt.title("Learning Rate = " + str(lr) + " ,Batch Size = " + str(batch_size) + " ,Epochs = "+str(epochs))
    plt.show()










    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, sampler=test_sampler, num_workers=2)
    testcount = 0
    correct = 0
    top3correct = 0

    top1error_plane = 0
    top1error_car = 0
    top1error_bird= 0
    top1error_cat=0
    top1error_deer=0
    top1error_dog=0
    top1error_frog=0
    top1error_horse=0
    top1error_ship=0
    top1error_truck=0
    top3error_plane = 0
    top3error_car = 0
    top3error_bird = 0
    top3error_cat = 0
    top3error_deer = 0
    top3error_dog = 0
    top3error_frog = 0
    top3error_horse = 0
    top3error_ship = 0
    top3error_truck = 0
    print("########################## TESTING DATA TOP 1 ##################################")

    all_preds = []
    all_labels = []
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
            top3correct +=1

        # TOP1 error rate aka percentage of time it was NOT correct
        if predicted != int(labels):
            if predicted == 0 :
                top1error_plane += 1
            elif predicted == 1 :
                top1error_car += 1
            elif predicted == 2:
                top1error_bird += 1
            elif predicted == 3:
                top1error_cat +=1
            elif predicted == 4:
                top1error_deer +=1
            elif predicted == 5:
                top1error_dog +=1
            elif predicted == 6:
                top1error_frog += 1
            elif predicted == 7:
                top1error_horse += 1
            elif predicted == 8:
                top1error_ship += 1
            elif predicted == 9:
                top1error_truck += 1

        # TOP3 error rate aka percentage of time it was NOT correct
        if 0 not in top3[0] and int(labels) == 0:
            top3error_plane += 1
        elif 1 not in top3[0] and int(labels) == 1:
            top3error_car += 1
        elif 2 not in top3[0] and int(labels) == 2:
            top3error_bird += 1
        elif 3 not in top3[0] and int(labels) == 3:
            top3error_cat += 1
        elif 4 not in top3[0] and int(labels) == 4:
            top3error_deer += 1
        elif 5 not in top3[0] and int(labels) == 5:
            top3error_dog += 1
        elif 6 not in top3[0] and int(labels) == 6:
            top3error_frog += 1
        elif 7 not in top3[0] and int(labels) == 7:
            top3error_horse += 1
        elif 8 not in top3[0] and int(labels) == 8:
            top3error_ship += 1
        elif 9 not in top3[0] and int(labels) == 9:
            top3error_truck += 1
    all_preds = torch.FloatTensor(all_preds)
    all_labels = torch.FloatTensor(all_labels)
    stacked = torch.stack((all_labels, all_preds), dim=1)

    stacked[0].tolist()
    cmt = torch.zeros(10, 10, dtype=torch.int64)

    for p in stacked:
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1
    cmt = cmt.numpy()


    images = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]
    df_cm = pd.DataFrame(cmt, index=[i for i in images],
                         columns=[i for i in images])
    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Test Data Confusion Matrix")
    plt.show()


    time.sleep(2)

    print("Percentage of predictions that WERE correct:  ", round(100 * (correct / testcount),2))
    # TOP 1 ERROR RATE
    planer = 100 * (top1error_plane / testcount)
    carer = 100 * (top1error_car / testcount)
    birder = 100 * (top1error_bird / testcount)
    cater = 100 * (top1error_cat / testcount)
    deerer = 100 * (top1error_deer / testcount)
    doger = 100 * (top1error_dog / testcount)
    froger = 100 * (top1error_frog / testcount)
    horseer = 100 * (top1error_horse / testcount)
    shiper = 100 * (top1error_ship / testcount)
    trucker = 100 * (top1error_truck / testcount)
    print("Percentage of predictions that were NOT correct, plane :  ", round(planer,2))
    print("Percentage of predictions that were NOT correct, car :  ", round(carer,2))
    print("Percentage of predictions that were NOT correct, bird :  ", round(birder,2))
    print("Percentage of predictions that were NOT correct, cat :  ", round(cater,2))
    print("Percentage of predictions that were NOT correct, deer :  ", round(deerer,2))
    print("Percentage of predictions that were NOT correct, dog :  ", round(doger,2))
    print("Percentage of predictions that were NOT correct, frog :  ", round(froger,2))
    print("Percentage of predictions that were NOT correct, horse :  ", round(horseer,2))
    print("Percentage of predictions that were NOT correct, ship :  ", round(shiper,2))
    print("Percentage of predictions that were NOT correct, truck :  ", round(trucker,2))
    print('Sum of top 1 errors: ', round(sum([planer, carer, birder, cater, deerer, doger, froger, horseer, shiper, trucker]),2))
    print('Mean of top 1 errors: ', round((sum([planer, carer, birder, cater, deerer, doger, froger, horseer, shiper, trucker])/10),2))


    print("########################## TESTING DATA TOP 3 ##################################")

    print("Percentage of predictions that WERE correct (top 3):  ", round(100 * (top3correct / testcount),2))
    # TOP 1 ERROR RATE
    planer = 100 * (top3error_plane / testcount)
    carer = 100 * (top3error_car / testcount)
    birder = 100 * (top3error_bird / testcount)
    cater = 100 * (top3error_cat / testcount)
    deerer = 100 * (top3error_deer / testcount)
    doger = 100 * (top3error_dog / testcount)
    froger = 100 * (top3error_frog / testcount)
    horseer = 100 * (top3error_horse / testcount)
    shiper = 100 * (top3error_ship / testcount)
    trucker = 100 * (top3error_truck / testcount)
    print("Percentage of predictions that were NOT correct, plane :  ", round(planer,2))
    print("Percentage of predictions that were NOT correct, car :  ", round(carer,2))
    print("Percentage of predictions that were NOT correct, bird :  ", round(birder,2))
    print("Percentage of predictions that were NOT correct, cat :  ", round(cater,2))
    print("Percentage of predictions that were NOT correct, deer :  ", round(deerer,2))
    print("Percentage of predictions that were NOT correct, dog :  ", round(doger,2))
    print("Percentage of predictions that were NOT correct, frog :  ", round(froger,2))
    print("Percentage of predictions that were NOT correct, horse :  ", round(horseer,2))
    print("Percentage of predictions that were NOT correct, ship :  ", round(shiper,2))
    print("Percentage of predictions that were NOT correct, truck :  ", round(trucker,2))
    print('Sum of top 3 errors: ', round(sum([planer, carer, birder, cater, deerer, doger, froger, horseer, shiper, trucker]),2))
    print('Mean of top 3 errors: ',
          round((sum([planer, carer, birder, cater, deerer, doger, froger, horseer, shiper, trucker]) / 10),2))







    ####################################### training  ####################################################
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, sampler=train_sampler, num_workers=2)
    traincount = 0
    correct = 0
    top3correctTrain = 0

    top1error_plane = 0
    top1error_car = 0
    top1error_bird= 0
    top1error_cat=0
    top1error_deer=0
    top1error_dog=0
    top1error_frog=0
    top1error_horse=0
    top1error_ship=0
    top1error_truck=0

    top3error_plane = 0
    top3error_car = 0
    top3error_bird = 0
    top3error_cat = 0
    top3error_deer = 0
    top3error_dog = 0
    top3error_frog = 0
    top3error_horse = 0
    top3error_ship = 0
    top3error_truck = 0
    print("########################## TRAINING DATA TOP 1 ##################################")
    all_preds_train = []
    all_labels_train = []

    for inputs, labels in train_loader:
        inputs, labels = Variable(inputs), Variable(labels)

        TrainO = net(inputs)
        _, predicted = torch.max(TrainO.data, 1)
        all_preds_train.append(int(predicted))
        all_labels_train.append(int(labels))
        top3 = torch.topk(TrainO.data, 3)
        top3 = top3.indices.tolist()

        # TOP1 accuracy
        if predicted == int(labels):
            correct += 1
        traincount += 1
        # TOP3 accuracy
        if int(labels) in top3[0]:
            top3correctTrain += 1


        # TOP1 error rate aka percentage of time it was NOT correct
        if predicted != int(labels):
            if predicted == 0 :
                top1error_plane += 1
            elif predicted == 1 :
                top1error_car += 1
            elif predicted == 2:
                top1error_bird += 1
            elif predicted == 3:
                top1error_cat +=1
            elif predicted == 4:
                top1error_deer +=1
            elif predicted == 5:
                top1error_dog +=1
            elif predicted == 6:
                top1error_frog += 1
            elif predicted == 7:
                top1error_horse += 1
            elif predicted == 8:
                top1error_ship += 1
            elif predicted == 9:
                top1error_truck += 1


            # TOP3 error rate aka percentage of time it was NOT correct
            if 0 not in top3[0] and int(labels) == 0:
                top3error_plane += 1
            elif 1 not in top3[0] and int(labels) == 1:
                top3error_car += 1
            elif 2 not in top3[0] and int(labels) == 2:
                top3error_bird += 1
            elif 3 not in top3[0] and int(labels) == 3:
                top3error_cat += 1
            elif 4 not in top3[0] and int(labels) == 4:
                top3error_deer += 1
            elif 5 not in top3[0] and int(labels) == 5:
                top3error_dog += 1
            elif 6 not in top3[0] and int(labels) == 6:
                top3error_frog += 1
            elif 7 not in top3[0] and int(labels) == 7:
                top3error_horse += 1
            elif 8 not in top3[0] and int(labels) == 8:
                top3error_ship += 1
            elif 9 not in top3[0] and int(labels) == 9:
                top3error_truck += 1
    all_preds_train = torch.FloatTensor(all_preds)
    all_labels_train = torch.FloatTensor(all_labels)
    stacked = torch.stack((all_labels_train, all_preds_train), dim=1)

    stacked[0].tolist()
    cmt = torch.zeros(10, 10, dtype=torch.int64)

    for p in stacked:
        tl, pl = p.tolist()
        cmt[int(tl), int(pl)] = cmt[int(tl), int(pl)] + 1
    cmt = cmt.numpy()


    images = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    df_cm = pd.DataFrame(cmt, index=[i for i in images],
                         columns=[i for i in images])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Training Data Confusion Matrix")
    plt.show()
    time.sleep(1)


    print("Percentage of predictions that WERE correct:  ", round(100 * (correct / traincount),2))
    # TOP 1 ERROR RATE
    planer = 100 * (top1error_plane / traincount)
    carer = 100 * (top1error_car / traincount)
    birder = 100 * (top1error_bird / traincount)
    cater = 100 * (top1error_cat / traincount)
    deerer = 100 * (top1error_deer / traincount)
    doger = 100 * (top1error_dog / traincount)
    froger = 100 * (top1error_frog / traincount)
    horseer = 100 * (top1error_horse / traincount)
    shiper = 100 * (top1error_ship / traincount)
    trucker = 100 * (top1error_truck / traincount)
    print("Percentage of predictions that were NOT correct, plane :  ", round(planer, 2))
    print("Percentage of predictions that were NOT correct, car :  ", round(carer, 2))
    print("Percentage of predictions that were NOT correct, bird :  ", round(birder, 2))
    print("Percentage of predictions that were NOT correct, cat :  ", round(cater, 2))
    print("Percentage of predictions that were NOT correct, deer :  ", round(deerer, 2))
    print("Percentage of predictions that were NOT correct, dog :  ", round(doger, 2))
    print("Percentage of predictions that were NOT correct, frog :  ", round(froger, 2))
    print("Percentage of predictions that were NOT correct, horse :  ", round(horseer, 2))
    print("Percentage of predictions that were NOT correct, ship :  ", round(shiper, 2))
    print("Percentage of predictions that were NOT correct, truck :  ", round(trucker, 2))
    print('Sum of top 1 errors: ',
          round(sum([planer, carer, birder, cater, deerer, doger, froger, horseer, shiper, trucker]), 2))
    print('Mean of top 1 errors: ',
          round((sum([planer, carer, birder, cater, deerer, doger, froger, horseer, shiper, trucker]) / 10), 2))

    print("########################## TRAINING DATA TOP 3 ##################################")

    print("Percentage of predictions that WERE correct (top 3):  ", round(100 * (top3correctTrain / traincount),2))
    # TOP 1 ERROR RATE
    planer = 100 * (top3error_plane / traincount)
    carer = 100 * (top3error_car / traincount)
    birder = 100 * (top3error_bird / traincount)
    cater = 100 * (top3error_cat / traincount)
    deerer = 100 * (top3error_deer / traincount)
    doger = 100 * (top3error_dog / traincount)
    froger = 100 * (top3error_frog / traincount)
    horseer = 100 * (top3error_horse / traincount)
    shiper = 100 * (top3error_ship / traincount)
    trucker = 100 * (top3error_truck / traincount)
    print("Percentage of predictions that were NOT correct, plane :  ", round(planer, 2))
    print("Percentage of predictions that were NOT correct, car :  ", round(carer, 2))
    print("Percentage of predictions that were NOT correct, bird :  ", round(birder, 2))
    print("Percentage of predictions that were NOT correct, cat :  ", round(cater, 2))
    print("Percentage of predictions that were NOT correct, deer :  ", round(deerer, 2))
    print("Percentage of predictions that were NOT correct, dog :  ", round(doger, 2))
    print("Percentage of predictions that were NOT correct, frog :  ", round(froger, 2))
    print("Percentage of predictions that were NOT correct, horse :  ", round(horseer, 2))
    print("Percentage of predictions that were NOT correct, ship :  ", round(shiper, 2))
    print("Percentage of predictions that were NOT correct, truck :  ", round(trucker, 2))
    print('Sum of top 3 errors: ',
          round(sum([planer, carer, birder, cater, deerer, doger, froger, horseer, shiper, trucker]), 2))
    print('Mean of top 3 errors: ',
          round((sum([planer, carer, birder, cater, deerer, doger, froger, horseer, shiper, trucker]) / 10), 2))




