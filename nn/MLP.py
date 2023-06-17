import torch
import torch.nn as nn
from dataset import CutCellData, CutInfData
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt

class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(input_dim, attention_dim)
        self.context_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        attention_weights = self.context_vector(torch.tanh(self.linear(x)))
        attention_weights = torch.softmax(attention_weights, dim=1)
        return attention_weights

class MLPWithAttention(nn.Module):
    def __init__(self, node_feature_dim, cut_feature_dim, cell_feature_dim, hidden_dim, output_dim):
        super(MLPWithAttention, self).__init__()

        self.node_attention = Attention(node_feature_dim, hidden_dim)
        self.cut_attention = Attention(cut_feature_dim, hidden_dim)
        self.cell_attention = Attention(cell_feature_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(node_feature_dim + cut_feature_dim + cell_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, node_features, cut_features, cell_features):
        node_attention_weights = self.node_attention(node_features)
        cut_attention_weights = self.cut_attention(cut_features)
        cell_attention_weights = self.cell_attention(cell_features)

        node_features_weighted = torch.sum(node_attention_weights * node_features, dim=1)
        cut_features_weighted = torch.sum(cut_attention_weights * cut_features, dim=1)
        cell_features_weighted = torch.sum(cell_attention_weights * cell_features, dim=1)

        combined_features = torch.cat([node_features_weighted, cut_features_weighted, cell_features_weighted], dim=1)

        output = self.mlp(combined_features)

        return output

def inference_data(model, inf_loader, inf_dataset, stdD, meanD):
    model.eval()  # set the model to evaluation model
    predictions = []
    with torch.no_grad():
        for i, (node_features, cut_features, cell_features) in enumerate(inf_loader):
            node_features = node_features.unsqueeze(1)
            cut_features = cut_features.unsqueeze(1)
            cell_features = cell_features.unsqueeze(1)

            outputs = model(node_features, cut_features, cell_features)
            # Convert predictions back to the original scale
            outputs = outputs * stdD + meanD
            predictions.extend(outputs.numpy().flatten().tolist())


    # Write the predictions to a file
    # with open('predictions.txt', 'w') as f:
    #     for pred in predictions:
    #         f.write(f"{pred}\n")
    with open('../data/test/predictions.txt', 'w') as f:
        for i in range(len(inf_dataset)):
            lineStr = str(inf_dataset.cut_data['node_id'][i])+ ',' + inf_dataset.cut_data['cell_name'][i] + ',' + str(inf_dataset.cut_data['phase'][i]) + ',' + "{:.4f}".format(predictions[i]) + '\n'
            # f.write(f"{inf_dataset.cut_data['node_id'][i],inf_dataset.cut_data['cell_name'][i],inf_dataset.cut_data['phase'][i],predictions[i]}\n")
            f.write(lineStr)

    return predictions

if __name__ == '__main__':
    node_file = "../data/train/adder_node_emb.csv"
    cut_file = '../data/train/adder_cut_emb.csv'
    cell_file = '../data/train/adder_cell_emb.csv'
    labels_file = '../data/train/adder_lables.csv'


    # Instantiate CutCellData
    cut_cell_data = CutCellData(node_file, cut_file, cell_file, labels_file)

    # Set batch size
    batch_size = 32
    # Assuming you have dataset as instance of your CutCellData
    dataset = CutCellData(node_file, cut_file, cell_file, labels_file)
    stdD, meanD = dataset.std, dataset.mean
    #
    # Define the split sizes
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # rest for testing

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Now you can create data loaders for each of these
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the MLPWithAttention model
    node_feature_dim = 10
    cut_feature_dim = 61
    cell_feature_dim = 56
    hidden_dim = 32
    output_dim = 1

    model = MLPWithAttention(node_feature_dim, cut_feature_dim, cell_feature_dim, hidden_dim, output_dim)
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)

    # Training loop
    num_epochs = 100
    best_loss = float('inf')
    losses = []  # to keep track of losses

    for epoch in range(num_epochs):
        for i, (node_features, cut_features, cell_features, labels) in enumerate(train_loader):
            # Reshape input features for the attention mechanism
            node_features = node_features.unsqueeze(1)
            cut_features = cut_features.unsqueeze(1)
            cell_features = cell_features.unsqueeze(1)

            # Forward pass
            outputs = model(node_features, cut_features, cell_features)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        losses.append(loss.item())  # add the loss of each epoch to the list

        # save the best model .
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_losses = []
                for node_features, cut_features, cell_features, labels in test_loader:
                    node_features = node_features.unsqueeze(1)
                    cut_features = cut_features.unsqueeze(1)
                    cell_features = cell_features.unsqueeze(1)

                    outputs = model(node_features, cut_features, cell_features)
                    loss = criterion(outputs, labels)
                    test_losses.append(loss.item())
                avg_test_loss = sum(test_losses) / len(test_losses)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss}")

                # Save the model if it has the best loss so far
                if avg_test_loss < best_loss:
                    best_loss = avg_test_loss
                    torch.save(model.state_dict(), '../data/best_model.pt')


    # After training, plot the loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()

    # model.load_state_dict(torch.load('../data/best_model.pt'))
    # inf_dataset = CutInfData(node_file, cut_file, cell_file)
    # inf_loader = DataLoader(inf_dataset, batch_size=batch_size, shuffle=False)
    #
    # inference_data(model, inf_loader, inf_dataset, stdD, meanD)