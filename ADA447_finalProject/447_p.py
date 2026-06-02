import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 0. REPRODUCIBILITY
# =====================================================================
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. SYNTHETIC AIRCRAFT BALANCE DATA GENERATION
# =====================================================================
print("Simulating virtual aircraft balance data...")

num_samples = 2000
seq_length = 10

# Physical limit of the lead screw mechanism
max_speed = 0.5  # maximum counterweight movement per time step
current_weight_pos = 0.0

time = np.arange(num_samples)

# Noisy pitch angle signal: aircraft motion + vibration/noise
pitch_angle = (
    10 * np.sin(time / 20)
    + 3 * np.sin(time / 7)
    + np.random.normal(0, 1.5, num_samples)
)

# Pitch velocity: trend information
pitch_velocity = np.gradient(pitch_angle)

ideal_positions = []
limited_positions = []

for i in range(num_samples):
    # Mathematical ideal position.
    # This is what pure math wants, but it may be physically unreachable.
    ideal_pos = -0.30 * pitch_angle[i] - 0.08 * pitch_velocity[i]

    # Lead screw motor cannot teleport the counterweight.
    distance_to_travel = ideal_pos - current_weight_pos

    if distance_to_travel > max_speed:
        actual_movement = max_speed
    elif distance_to_travel < -max_speed:
        actual_movement = -max_speed
    else:
        actual_movement = distance_to_travel

    current_weight_pos += actual_movement

    ideal_positions.append(ideal_pos)
    limited_positions.append(current_weight_pos)

ideal_positions = np.array(ideal_positions)
limited_positions = np.array(limited_positions)

# Input features:
# 1. pitch angle
# 2. pitch velocity
# 3. current counterweight position
X_data = np.column_stack((pitch_angle, pitch_velocity, limited_positions))


# Target:
# We train the model to predict the mathematical ideal position.
# The physics-informed model is punished if it tries to reach this target
# with an impossible motor movement.
Y_data = ideal_positions

# =====================================================================
# 2. CREATE SEQUENCES FOR LSTM
# =====================================================================
X_seq, Y_seq, physical_target_seq = [], [], []

for i in range(len(X_data) - seq_length):
    X_seq.append(X_data[i:i + seq_length])
    Y_seq.append(Y_data[i + seq_length])

    # This is the physically reachable position at the next time step.
    # We use it only for plotting and interpretation.
    physical_target_seq.append(limited_positions[i + seq_length])

X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32).unsqueeze(-1)
physical_target_tensor = torch.tensor(np.array(physical_target_seq), dtype=torch.float32).unsqueeze(-1)

# Train-test split without shuffling because this is time-series data
train_size = int(len(X_tensor) * 0.8)

X_train = X_tensor[:train_size]
Y_train = Y_tensor[:train_size]

X_test = X_tensor[train_size:]
Y_test = Y_tensor[train_size:]

physical_target_test = physical_target_tensor[train_size:]

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# =====================================================================
# 3. PHYSICS PENALTY MATRIX FOR ATTENTION
# =====================================================================
def create_attention_physics_penalty(x, max_speed=0.5, penalty_scale=4.0):
    """
    Creates a physics-based penalty matrix for the Attention mechanism.

    x shape: (batch, seq_length, num_features)

    Feature indexes:
    0 -> pitch angle
    1 -> pitch velocity
    2 -> current counterweight position

    Idea:
    Attention compares every time step with every other time step.
    If the counterweight position difference between two time steps is
    larger than what the motor could physically travel in that time gap,
    we penalize that attention connection.
    """

    weight_pos = x[:, :, 2]  # (batch, seq_length)

    batch_size, sequence_length = weight_pos.shape

    # Pairwise position difference
    position_diff = torch.abs(
        weight_pos.unsqueeze(2) - weight_pos.unsqueeze(1)
    )  # (batch, seq_length, seq_length)

    # Pairwise time distance
    time_index = torch.arange(sequence_length, device=x.device).float()
    time_distance = torch.abs(
        time_index.unsqueeze(0) - time_index.unsqueeze(1)
    )  # (seq_length, seq_length)

    # Maximum physically possible movement between two time steps
    allowed_movement = max_speed * time_distance
    allowed_movement = allowed_movement.unsqueeze(0)  # (1, seq_length, seq_length)

    # Penalize only physically impossible transitions
    violation = torch.clamp(position_diff - allowed_movement, min=0.0)

    penalty_matrix = violation * penalty_scale

    return penalty_matrix


def speed_limit_loss(prediction, x, max_speed=0.5):
    """
    Output-level physics loss.

    Even if the mathematical target is far away, the predicted next position
    should not require a movement larger than the motor's speed limit.

    prediction shape: (batch, 1)
    x shape: (batch, seq_length, num_features)
    """

    current_weight_position = x[:, -1, 2].unsqueeze(-1)

    required_movement = torch.abs(prediction - current_weight_position)

    violation = torch.clamp(required_movement - max_speed, min=0.0)

    return torch.mean(violation ** 2)


# =====================================================================
# 4. PHYSICS-INFORMED ATTENTION MODULE
# =====================================================================
class PhysicsInformedAttention(nn.Module):
    def __init__(self, hidden_dim, apply_physics=True, penalty_lambda=1.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.apply_physics = apply_physics
        self.penalty_lambda = penalty_lambda

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, penalty_matrix):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Standard scaled dot-product attention score
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)

        # Physics-informed modification
        if self.apply_physics:
            scores = scores - self.penalty_lambda * penalty_matrix

        attention_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, V)

        return context, attention_weights


# =====================================================================
# 5. AIRCRAFT BALANCING MODEL
# =====================================================================
class AircraftBalancingModel(nn.Module):
    def __init__(self, num_sensors=3, hidden_dim=32, apply_physics=True):
        super().__init__()

        # 1D CNN: noise filter for sensor data
        self.noise_filter = nn.Conv1d(
            in_channels=num_sensors,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        # LSTM: memory module
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Attention: standard or physics-informed depending on apply_physics
        self.attention = PhysicsInformedAttention(
            hidden_dim=hidden_dim,
            apply_physics=apply_physics,
            penalty_lambda=1.0
        )

        # MLP output head
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x, penalty_matrix):
        # Conv1D expects: (batch, channels, time)
        x = x.transpose(1, 2)

        x = F.relu(self.noise_filter(x))

        # LSTM expects: (batch, time, features)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)

        context, attention_weights = self.attention(lstm_out, penalty_matrix)

        # Use the final time step representation
        out = context[:, -1, :]

        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out, attention_weights


# =====================================================================
# 6. TRAINING
# =====================================================================
standard_model = AircraftBalancingModel(apply_physics=False)
physics_model = AircraftBalancingModel(apply_physics=True)

criterion = nn.MSELoss()

opt_standard = optim.Adam(standard_model.parameters(), lr=0.005)
opt_physics = optim.Adam(physics_model.parameters(), lr=0.005)

epochs = 80
physics_loss_weight = 8.0

print("\nTraining models...")

for epoch in range(epochs):
    standard_model.train()
    physics_model.train()

    # Create physics penalty matrix from actual input sequence
    train_penalty_matrix = create_attention_physics_penalty(
        X_train,
        max_speed=max_speed,
        penalty_scale=4.0
    )

    # -----------------------------
    # Standard model
    # -----------------------------
    opt_standard.zero_grad()

    pred_std, _ = standard_model(X_train, train_penalty_matrix)

    loss_std = criterion(pred_std, Y_train)

    loss_std.backward()
    opt_standard.step()

    # -----------------------------
    # Physics-informed model
    # -----------------------------
    opt_physics.zero_grad()

    pred_phy, _ = physics_model(X_train, train_penalty_matrix)

    mse_phy = criterion(pred_phy, Y_train)

    physical_loss = speed_limit_loss(
        pred_phy,
        X_train,
        max_speed=max_speed
    )

    loss_phy = mse_phy + physics_loss_weight * physical_loss

    loss_phy.backward()
    opt_physics.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Standard MSE: {loss_std.item():.4f} | "
            f"Physics MSE: {mse_phy.item():.4f} | "
            f"Physics Penalty: {physical_loss.item():.4f}"
        )


# =====================================================================
# 7. EVALUATION METRICS
# =====================================================================
def count_speed_violations(predictions, x, max_speed=0.5):
    """
    Counts how many predictions require the counterweight to move faster
    than the motor's physical speed limit.
    """

    current_weight_position = x[:, -1, 2].detach().cpu().numpy().reshape(-1)

    predictions = predictions.reshape(-1)

    required_movement = np.abs(predictions - current_weight_position)

    violations = np.sum(required_movement > max_speed)

    return int(violations)


def mean_required_movement(predictions, x):
    current_weight_position = x[:, -1, 2].detach().cpu().numpy().reshape(-1)

    predictions = predictions.reshape(-1)

    return np.mean(np.abs(predictions - current_weight_position))


def smoothness_score(predictions):
    """
    Lower value means smoother prediction curve.
    """
    predictions = predictions.reshape(-1)
    return np.mean(np.abs(np.diff(predictions)))


standard_model.eval()
physics_model.eval()

test_penalty_matrix = create_attention_physics_penalty(
    X_test,
    max_speed=max_speed,
    penalty_scale=4.0
)

with torch.no_grad():
    std_predictions, std_attention = standard_model(X_test, test_penalty_matrix)
    phy_predictions, phy_attention = physics_model(X_test, test_penalty_matrix)

std_predictions_np = std_predictions.detach().cpu().numpy()
phy_predictions_np = phy_predictions.detach().cpu().numpy()

actual_ideal_np = Y_test.detach().cpu().numpy()
physical_target_np = physical_target_test.detach().cpu().numpy()

std_mse = np.mean((std_predictions_np - actual_ideal_np) ** 2)
phy_mse = np.mean((phy_predictions_np - actual_ideal_np) ** 2)

std_violations = count_speed_violations(std_predictions_np, X_test, max_speed=max_speed)
phy_violations = count_speed_violations(phy_predictions_np, X_test, max_speed=max_speed)

std_mean_move = mean_required_movement(std_predictions_np, X_test)
phy_mean_move = mean_required_movement(phy_predictions_np, X_test)

std_smoothness = smoothness_score(std_predictions_np)
phy_smoothness = smoothness_score(phy_predictions_np)

print("\n================ RESULTS ================")
print(f"Standard Model MSE: {std_mse:.4f}")
print(f"Physics-Informed Model MSE: {phy_mse:.4f}")
print("-----------------------------------------")
print(f"Standard Model Speed Violations: {std_violations}")
print(f"Physics-Informed Model Speed Violations: {phy_violations}")
print("-----------------------------------------")
print(f"Standard Model Mean Required Movement: {std_mean_move:.4f}")
print(f"Physics-Informed Model Mean Required Movement: {phy_mean_move:.4f}")
print("-----------------------------------------")
print(f"Standard Model Smoothness Score: {std_smoothness:.4f}")
print(f"Physics-Informed Model Smoothness Score: {phy_smoothness:.4f}")
print("=========================================")


# =====================================================================
# 8. VISUALIZATION
# =====================================================================
plt.figure(figsize=(14, 6))

plt.plot(
    actual_ideal_np[:150],
    label="Mathematical Ideal Target",
    color="black",
    linestyle="dashed",
    linewidth=2
)

plt.plot(
    physical_target_np[:150],
    label="Physically Reachable Target",
    color="gray",
    linestyle="dotted",
    linewidth=2
)

plt.plot(
    std_predictions_np[:150],
    label="Standard LSTM + Attention",
    color="red",
    alpha=0.8
)

plt.plot(
    phy_predictions_np[:150],
    label="Physics-Informed LSTM + Attention",
    color="green",
    linewidth=2
)

plt.title("Dynamic Balance Prediction with Physics-Informed Attention", fontsize=14)
plt.xlabel("Time Steps in Test Set", fontsize=12)
plt.ylabel("Counterweight Position", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# =====================================================================
# 9. ATTENTION VISUALIZATION FOR ONE SAMPLE
# =====================================================================
sample_index = 20

plt.figure(figsize=(8, 6))
plt.imshow(
    phy_attention[sample_index].detach().cpu().numpy(),
    aspect="auto"
)
plt.colorbar(label="Attention Weight")
plt.title("Physics-Informed Attention Weights for One Test Sample")
plt.xlabel("Key Time Step")
plt.ylabel("Query Time Step")
plt.tight_layout()
plt.show()