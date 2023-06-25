import torch
from dataset import CutCellData, CutInfData
from HAN import FusionModel
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import glob
import  numpy as np
import pickle


def inference_circuit(model, inf_loader, inf_dataset, circuit_name, epoch):
    model.eval()  # set the model to evaluation model
    predictions = []
    with torch.no_grad():
        for i, (node_features, cut_features, cell_features) in enumerate(inf_loader):
            outputs = model(node_features, cut_features, cell_features)
            predictions.extend(outputs.numpy().flatten().tolist())

    with open('../data/test/{}_predictions_{}.txt'.format(circuit_name, epoch), 'w') as f:
        for i in range(len(inf_dataset)):
            lineStr = str(inf_dataset.cut_data['node_id'][i])+ ',' + inf_dataset.cut_data['cell_name'][i] + ',' + \
                      str(inf_dataset.cut_data['cut_hashID'][i]) + ',' + str(inf_dataset.cut_data['phase'][i]) + ',' + "{:.4f}".format(predictions[i]) + '\n'
            f.write(lineStr)


if __name__ == '__main__':
    # node_files = sorted(glob.glob('../data/train_delay/*_node_emb.csv'))
    # cut_files = sorted(glob.glob('../data/train_delay/*_cut_emb.csv'))
    # cell_files = sorted(glob.glob('../data/train_delay/*_cell_emb.csv'))
    # labels_files = sorted(glob.glob('../data/train_delay/*_lables.csv'))

    node_files = sorted(glob.glob('../data/train_delay/i2c_node_emb.csv'))
    cut_files = sorted(glob.glob('../data/train_delay/i2c_cut_emb.csv'))
    cell_files = sorted(glob.glob('../data/train_delay/i2c_cell_emb.csv'))
    labels_files = sorted(glob.glob('../data/train_delay/i2c_lables.csv'))

    MAXLEVEL, MAXFANOUT, MAXROOTFANOUT = 9000, 1000, 1000
    LEVELDIM, FANOUTDIM, ROOTFANOUTDIM = 4, 4, 4
    levelEmb = nn.Embedding(MAXLEVEL, LEVELDIM)
    fanoutEmb = nn.Embedding(MAXFANOUT, FANOUTDIM)
    rootFanoutEmb = nn.Embedding(MAXROOTFANOUT, ROOTFANOUTDIM)

    # Instantiate CutCellData
    dataset = CutCellData(node_files, cut_files, cell_files, labels_files, levelEmb, fanoutEmb, rootFanoutEmb)

    # node_fea: 6x28, cut_fea: 1x16, cell_fea: 1x61

    # Set batch size
    batch_size = 32
    # Assuming you have dataset as instance of your CutCellData
    # dataset = CutCellData(node_files, cut_files, cell_files, labels_files)
    # stdD, meanD = dataset.std, dataset.mean
    #
    # Define the split sizes
    train_size = int(0.8 * len(dataset))   # 80% for training
    test_size = len(dataset) - train_size  # rest for testing

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Now you can create data loaders for each of these
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the MLPWithAttention model
    node_feature_dim = 28
    cut_feature_dim = 16
    cell_feature_dim = 60
    postmlp_dim = 32
    fusion_dim = 12
    output_dim = 1
    pdrop = 0.1

    # model = MLPWithAttention(node_feature_dim, cut_feature_dim, cell_feature_dim, hidden_dim, output_dim)
    # model = FusionModel(node_feature_dim, cut_feature_dim, cell_feature_dim, fusion_size, pdrop)
    model = FusionModel(node_feature_dim, cut_feature_dim, cell_feature_dim, fusion_dim, postmlp_dim, pdrop)
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)


    # Training loop
    num_epochs = 10
    best_loss = float('inf')
    losses = []  # to keep track of losses
    #
    # for epoch in range(num_epochs):
    #     totalloss = 0.0
    #     i = 0
    #     for i, (node_features, cut_features, cell_features, labels) in enumerate(train_loader):
    #         # Forward pass
    #         outputs = model(node_features, cut_features, cell_features)
    #         loss = criterion(outputs, labels)
    #
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         totalloss += loss.item()
    #
    #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {totalloss/i:.4f}")
    #     losses.append(loss.item())  # add the loss of each epoch to the list
    #
    #     # save the best model .
    #     if (epoch + 1) % 1 == 0:
    #         model.eval()
    #         with torch.no_grad():
    #             test_losses = []
    #             for node_features, cut_features, cell_features, labels in test_loader:
    #                 outputs = model(node_features, cut_features, cell_features)
    #                 loss = criterion(outputs, labels)
    #                 test_losses.append(loss.item())
    #             avg_test_loss = sum(test_losses) / len(test_losses)
    #             print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss}")
    #             # Save the model if it has the best loss so far
    #             # if avg_test_loss < best_loss:
    #             best_loss = avg_test_loss
    #             torch.save(model.state_dict(), '../data/best_model_{}.pt'.format(epoch))
    #             embDict = {}
    #             embDict['levelEmb'] = levelEmb
    #             embDict['fanoutEmb'] = fanoutEmb
    #             embDict['rootFanoutEmb'] = rootFanoutEmb
    #             with open("../data/emb_{}.pickle".format(epoch), "wb") as embFile:
    #                 pickle.dump(embDict, embFile)

    original_loss = 0
    epoch = 2
    with open("../data/emb_{}.pickle".format(epoch), "rb") as embFile:
        embDict = pickle.load(embFile)
    model.load_state_dict(torch.load('../data/best_model_{}.pt'.format(epoch)))
    model.eval()
    with torch.no_grad():
        test_losses = []
        for node_features, cut_features, cell_features, labels in test_loader:
            outputs = model(node_features, cut_features, cell_features)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            original_loss = loss.item()
        avg_test_loss = sum(test_losses) / len(test_losses)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss}")


    num_features = 10
    feature_importance = np.zeros(num_features)
    # Permute each feature and calculate the drop in loss
    for feature_idx in range(num_features):
        permuted_loader = []
        permuted_loss = 0
        for node_features, cut_features, cell_features, labels in test_loader:

            # Create a copy of the original batch data 32x6x28
            permuted_batch_data = node_features.clone()

            indices = torch.randperm( permuted_batch_data.size(1) )

            # shuffled_tensor = tensor[:, :, indices]
            tmp = permuted_batch_data[:, indices, feature_idx]

            node_features[:, indices, feature_idx] = tmp
            permuted_batch_data2 = cut_features.clone()
            permuted_batch_data3 = cell_features.clone()
            labels3 = labels.clone()
            # permuted_loader.append([permuted_batch_data, permuted_batch_data2, permuted_batch_data3, labels3])



            # Compute the loss with the permuted feature
            with torch.no_grad():

                # for node_features, cut_features, cell_features, labels in permuted_loader:
                outputs = model(node_features, cut_features, cell_features)
                loss = criterion(outputs, labels)

                permuted_loss += loss.item()



        feature_importance[feature_idx] = permuted_loss

    print(feature_importance)
    # Normalize feature importance values
    feature_importance /= np.sum(feature_importance)


    # Print the feature importance values
    for feature_idx, importance in enumerate(feature_importance):
        print(f"Feature {feature_idx + 1}: Importance = {importance}")








    # # After training, plot the loss over epochs
    # plt.figure(figsize=(10, 5))
    # plt.plot(losses)
    # plt.title("Training Loss over Epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.grid(True)
    # plt.savefig('loss_plot.png')
    # plt.show()


    '''
    # circuits = ['adder', 'max', 'sin', 'bar', 'router', 'i2c', 'priority']
    # circuits = ["s444_comb", "C6288", "s526_comb", "rc256b", "log2", "square", "s9234_1_comb", "adder", "rc64b", "C880", "sin",
    # "div", "hyp", "mul64-booth", "aes", "C7552", "max", "mul32-booth", "sqrt", "multiplier", "64b_mult", "bar",
    # "s5378_comb", "C5315"]
    # circuits = ["s444_comb", "C6288", "s526_comb", "rc256b","s9234_1_comb", "adder", "rc64b", "C880", "sin",
    # "mul64-booth", "aes", "C7552", "max", "mul32-booth", "sqrt", "multiplier", "64b_mult", "bar",
    # "s5378_comb", "C5315", "log2", "square", "div", "hyp"]

    circuits = [ "square", "div", "hyp"]

    # circuits = ["cavlc", "ctrl", "dec", "i2c", "int2float", "priority", "router"]


    # circuits = ['i2c']
    for circuit_name in circuits:
        print("inference circuit: {}".format(circuit_name))
        node_files = sorted(glob.glob('../data/test/{}_node_emb.csv'.format(circuit_name)))
        cut_files = sorted(glob.glob('../data/test/{}_cut_emb.csv'.format(circuit_name)))
        cell_files = sorted(glob.glob('../data/test/{}_cell_emb.csv'.format(circuit_name)))
        epoch = 2
        with open("../data/emb_{}.pickle".format(epoch), "rb") as embFile:
            embDict = pickle.load(embFile)
        model.load_state_dict(torch.load('../data/best_model_{}.pt'.format(epoch)))

        inf_dataset = CutInfData(node_files, cut_files, cell_files, embDict['levelEmb'], embDict['fanoutEmb'], embDict['rootFanoutEmb'], dataset.min_arr, dataset.max_arr)
        inf_loader = DataLoader(inf_dataset, batch_size=batch_size, shuffle=False)
        inference_circuit(model, inf_loader, inf_dataset, circuit_name, epoch)
'''


    '''
    Epoch [1/10], Loss: 912.8060
Epoch [1/10], Test Loss: 442.6503408578726
Epoch [2/10], Loss: 573.8267
Epoch [2/10], Test Loss: 357.61758868877706
Epoch [3/10], Loss: 481.0738
Epoch [3/10], Test Loss: 303.74356853778545
Epoch [4/10], Loss: 417.8749
Epoch [4/10], Test Loss: 455.14270782470703
Epoch [5/10], Loss: 355.2172
Epoch [5/10], Test Loss: 1142.1675201416015
Epoch [6/10], Loss: 385.0006
Epoch [6/10], Test Loss: 737.5146354675293
Epoch [7/10], Loss: 302.1590
Epoch [7/10], Test Loss: 1491.701126216008
Epoch [8/10], Loss: 286.9986
Epoch [8/10], Test Loss: 1593.0306782062237
Epoch [9/10], Loss: 281.8444
Epoch [9/10], Test Loss: 1519.0283362755408
Epoch [10/10], Loss: 255.6557
Epoch [10/10], Test Loss: 1082.961026763916
WireLoad = "none"  Gates =   2448 ( 12.3 %)   Cap =  0.9 ff (  2.2 %)   Area =     2877.51 ( 87.7 %)   Delay =  1510.98 ps  ( 42.4 %)               
abc 03> read_lib -v /home/liujunfeng/SLAP/ML-Mapper/abc/asap7_clean.lib; read /home/liujunfeng/benchmarks/random-arith/bar.aig;  map -P ../../data/test/bar_predictions_2.txt; stime;
Warning: Detected 2 multi-output gates (for example, "FAx1_ASAP7_75t_R").
Loaded predicted supergate delay from the file "../../data/test/bar_predictions_2.txt ".
WireLoad = "none"  Gates =   2264 ( 17.7 %)   Cap =  0.8 ff (  3.2 %)   Area =     2577.04 ( 82.3 %)   Delay =  1248.09 ps  ( 15.7 %)    
    '''