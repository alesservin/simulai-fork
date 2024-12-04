import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from simulai.models import Transformer

def pad_features(features, target_dim):
    if features.shape[1] < target_dim:
        padding = target_dim - features.shape[1]
        return np.hstack([features, np.zeros((features.shape[0], padding))])
    return features


# Detect device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data = np.loadtxt('data/Data_CHF_Zhao_2020_ATE-My_Numerical_Version.csv', delimiter=',', skiprows=1)  # Adjust the file path if needed

# Extract input features (X) and target (y)
X = data[:, :-1]  # All columns except the last (input features)
y = data[:, -1:]  # The last column (target)

# Scale the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Padding to ensure compatibility with num_heads
X_train = pad_features(X_train, 8)
X_test = pad_features(X_test, 8)

# Convert to PyTorch tensors and move to the device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Define the Transformer model
def create_transformer():
    num_heads = 2
    embed_dim = 8  # Choose an embed_dim divisible by num_heads (e.g., 8, 16, 32, etc.)
    hidden_dim = 16
    number_of_encoders = 2
    number_of_decoders = 2

    # Configuration for the encoder and decoder MLP layers
    encoder_mlp_config = {
        "layers_units": [hidden_dim, hidden_dim],  # Hidden layers
        "activations": "relu",
        "input_size": embed_dim,
        "output_size": embed_dim,
        "name": "encoder_mlp",
    }

    decoder_mlp_config = {
        "layers_units": [hidden_dim, hidden_dim],  # Hidden layers
        "activations": "relu",
        "input_size": embed_dim,
        "output_size": embed_dim,
        "name": "decoder_mlp",
    }

    # Instantiate the Transformer model
    transformer = Transformer(
        num_heads_encoder=num_heads,
        num_heads_decoder=num_heads,
        embed_dim_encoder=embed_dim,
        embed_dim_decoder=embed_dim,
        output_dim=1,  # Output dimension matches the target column
        encoder_activation="relu",
        decoder_activation="relu",
        encoder_mlp_layer_config=encoder_mlp_config,
        decoder_mlp_layer_config=decoder_mlp_config,
        number_of_encoders=number_of_encoders,
        number_of_decoders=number_of_decoders,
        devices=device,
    )

    return transformer

# Initialize the Transformer model and move to the device
model = create_transformer().to(device)

# Use PyTorch optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Loss function
loss_function = torch.nn.MSELoss()

# Training loop
n_epochs = 5000
batch_size = 512
num_batches = int(np.ceil(len(X_train_tensor) / batch_size))

for epoch in range(n_epochs):
    epoch_loss = 0.0
    for i in range(num_batches):
        # Create mini-batches
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_train_tensor))
        X_batch = X_train_tensor[start:end]
        y_batch = y_train_tensor[start:end]

        # Forward pass
        predictions = model.forward(X_batch)
        loss = loss_function(predictions, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Logging every 100 epochs
    if epoch % 100 == 0:
        with torch.no_grad():
            test_predictions = model.forward(X_test_tensor)
            test_loss = loss_function(test_predictions, y_test_tensor).item()
        print(f"Epoch {epoch}/{n_epochs}, Train Loss: {epoch_loss/num_batches:.6f}, Test Loss: {test_loss:.6f}")

# Evaluate on test data (explicitly passing input data)
with torch.no_grad():
    test_predictions = model.forward(X_test_tensor)
    test_loss = loss_function(test_predictions, y_test_tensor).item()
    print(f"Final Test Loss: {test_loss:.6f}")

# Save the model
torch.save(model.state_dict(), "critical_heat_flux_transformer.pth")

# Save scalers
import joblib
joblib.dump(scaler_X, 'critical_heat_flux_transformer-scaler_X.pkl')
joblib.dump(scaler_y, 'critical_heat_flux_transformer-scaler_y.pkl')
