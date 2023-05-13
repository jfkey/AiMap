import torch
import torch.nn as nn
import torch.optim as optim


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_features, targets, target_lengths = batch
        input_features, targets = input_features.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(input_features, target_lengths)
        loss = nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_features, targets, target_lengths = batch
            input_features, targets = input_features.to(device), targets.to(device)

            output = model(input_features, target_lengths)
            loss = nn.CrossEntropyLoss()(output, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def calculate_accuracy(output, targets):
    _, predictions = torch.max(output, 1)
    correct = (predictions == targets).float()
    return correct.sum() / len(correct)



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate model
input_dim = 15
hidden_dim = 128
n_layers = 2
dropout = 0.5

model = PointerNet(input_dim, hidden_dim, n_layers, dropout).to(device)

# Set hyperparameters
learning_rate = 0.001
num_epochs = 50

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Instantiate dataset and dataloader
dataset = CutSelectionDataset(your_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train and evaluate model
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, dataloader, optimizer, device)
    val_loss = evaluate(model, dataloader, device)
    print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# Save model
torch.save(model.state_dict(), "model.pt")

# Load model
loaded_model = PointerNet(input_dim, hidden_dim, n_layers, dropout).to(device)
loaded_model.load_state_dict(torch.load("model.pt"))