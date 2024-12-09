import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler


def print(x):
    return open("lstm.log", "a").write(f"{x}\n")


def load_dataset(year: int = 2000, ticker: str = "GC=F"):
    data = yf.download(ticker, start=f"{year}-01-01")

    data = data.reset_index()
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
    data = data.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    data.columns = data.columns.droplevel(1)

    data.to_csv("data/data.csv", index=False)

    return "data/data.csv"


def run_model(data_file: str = "data/data.csv"):
    dataset_train = pd.read_csv(data_file)
    train_set = dataset_train.iloc[:, [1]].values  # selecting oepn column
    sc = MinMaxScaler(feature_range=(0, 1))
    train_set_scaled = sc.fit_transform(train_set)

    def create_sequences(data, seq_length):
        """
        Function to create sequences of data (since we are using an RNN), so at each time step, we will input a sequence of data
        Note that for now we care about predicting the 'open' price
        """
        sequences = []
        labels = []
        for i in range(len(data) - seq_length):
            seq = data[i: i + seq_length]
            label = data[
                i + seq_length, 0
            ]  # choosing the 'open' price as the label we want to predict (i.e. the next day's open price)
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    sequence_length = 120  # number of days to look back to predict the next day's open price (i.e. 120 days)

    X_train, y_train = create_sequences(train_set_scaled, sequence_length)

    # Convert the data to PyTorch tensors
    X_train_tensors = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensors = torch.tensor(y_train, dtype=torch.float32)

    print(f"X_train shape: {X_train_tensors.shape}")
    print(f"y_train shape: {y_train_tensors.shape}")

    class GoldLSTM(nn.Module):
        def __init__(
            self, input_size, hidden_size_LSTM, num_layers_LSTM, output_size=1
        ):
            super(GoldLSTM, self).__init__()
            # define my parameters
            self.input_size = input_size
            self.hidden_size_LSTM = hidden_size_LSTM
            self.num_layers_LSTM = num_layers_LSTM
            self.output_size = output_size

            # define our layers
            self.LSTM_1 = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size_LSTM,
                num_layers=num_layers_LSTM,
                bias=True,
                batch_first=True,
                dropout=0.2,
            )
            self.output_layer = nn.Linear(
                in_features=hidden_size_LSTM, out_features=output_size, bias=True
            )

        def forward(self, x):
            h_0 = torch.zeros(
                self.num_layers_LSTM, x.size(0), self.hidden_size_LSTM
            ).to(
                x.device
            )  # shape(num_layers,batch_size,hidden_size)

            c_0 = torch.zeros(
                self.num_layers_LSTM, x.size(0), self.hidden_size_LSTM
            ).to(
                x.device
            )  # shape: (num_layers, batch_size, hidden_size)

            out, (h_n, c_n) = self.LSTM_1(
                x, (h_0, c_0)
            )  # notice that we don't care about h_n here

            out = out[
                :, -1, :
            ]  # only care about the output of the last time step not all!!!

            out = self.output_layer(out)
            # no activation function on the last layer
            return out

    input_size = 1  # open feature
    hidden_size = 50  # Number of hidden units inside the RNN(or LSTM)
    num_layers = 4  # Number of LSTM layers (stacked on top of each other)
    output_size = 1  # 1 output neuron to predict the next day's open price
    batch_size = 32  # batch size

    model = GoldLSTM(input_size, hidden_size, num_layers, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create data loaders (without shuffling to preserve order)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )

    num_epochs = 150
    train_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X_batch)
            outputs = (outputs.squeeze())  # Remove unnecessary dimensions for comparison with y_batch

            # Compute the loss
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            running_train_loss += loss.item() * X_batch.size(0)

        # Average training loss for the epoch
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}")

    # Show training loss curve
    plt.plot(train_losses, label="Training Loss")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("images/plots/loss.png")

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        predictions = model(X_train_tensor).reshape(-1, 1)

    predictions_with_dummy = np.zeros(
        (predictions.shape[0], 2)
    )  # Create a (1292, 2) array with zeros
    predictions_with_dummy[:, 0] = predictions[
        :, 0
    ]  # Place predictions in the first column

    # Inverse transform using the scaler
    predictions_original_scale = sc.inverse_transform(predictions_with_dummy)[
        :, 0
    ]  # Inverse transform and keep only the first column

    # Do the same for the true values
    y_train_with_dummy = np.zeros(
        (y_train_tensor.shape[0], 2)
    )  # Create a (1292, 2) array with zeros
    y_train_with_dummy[:, 0] = y_train_tensor.reshape(
        -1
    )  # Place true values in the first column

    # Inverse transform and keep only the first column
    y_train_original_scale = sc.inverse_transform(y_train_with_dummy)[:, 0]

    # Show the true values vs. the predictions
    plt.plot(y_train_original_scale, label="True Values", color="b")
    plt.plot(predictions_original_scale, label="Predictions", color="r")
    plt.title("True Values vs Predictions")
    plt.xlabel("Time Steps")
    plt.ylabel("Price (Original Scale)")
    plt.legend()
    plt.savefig("images/plots/predictions_LSTM.png")

    FINAL_PREDICTION = predictions_original_scale[-1]
    open("prediction_LSTM.log", "w").write(str(FINAL_PREDICTION))
