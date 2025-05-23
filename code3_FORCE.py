import torch
import numpy as np
import matplotlib.pyplot as plt
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import surrogate
import torch.nn as nn

torch.manual_seed(0)

# Generate sine wave
t_eval = np.linspace(0, 50, 1000)
sin_wave = np.sin(t_eval)
data = torch.tensor(sin_wave, dtype=torch.float32).unsqueeze(1)  # [T, 1]

# Network parameters
num_hidden = 2000
num_outputs = 1
beta = 0.88

class SelfEvolvingSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias_layer = nn.Linear(1, num_hidden, bias=False)
        self.recurrent = nn.Linear(num_hidden, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        # Readout weights will be trained manually via FORCE
        self.readout_weights = torch.randn(num_hidden, num_outputs)  # shape [hidden, output]

        with torch.no_grad():
            base_value = 0.5
            noise_std = 0.1
            self.bias_layer.weight.data = torch.randn_like(self.bias_layer.weight) * noise_std + base_value
        self.bias_layer.weight.requires_grad = False
        self.recurrent.weight.requires_grad = False

    def forward(self, spikes_prev, mem, bias_value=0.0):
        bias_input = torch.ones((1, 1)) * bias_value
        bias_current = self.bias_layer(bias_input)
        recurrent_current = self.recurrent(spikes_prev)
        h = bias_current + recurrent_current
        spikes, mem = self.lif1(h, mem)

        # Manual readout
        out = torch.matmul(spikes, self.readout_weights)
        return spikes, mem, out

# Initialize model
model = SelfEvolvingSNN()

# FORCE learning parameters
alpha = 0.5
P = (1.0 / alpha) * torch.eye(num_hidden)  # Inverse correlation matrix

# Training
epochs = 1
time_steps = 1000
bias_value = 0.5

for epoch in range(epochs):
    spikes = torch.zeros((1, num_hidden))
    mem = torch.zeros_like(spikes)
    for t in range(time_steps):
        target = data[t].unsqueeze(0)  # [1, 1]
        spikes, mem, output = model(spikes, mem, bias_value=bias_value)

        #Apply FORCE
        if t % 1 == 0:
            r = spikes.squeeze(0).unsqueeze(1)  # [hidden, 1]
            k = torch.matmul(P, r)
            rPr = torch.matmul(r.T, k)
            c = 1.0 / (1.0 + rPr)
            P -= c * torch.matmul(k, k.T)

            e = output - target  # [1, 1]
            dw = -e.item() * c * k
            model.readout_weights += dw

    print(f"Epoch {epoch+1}/{epochs} complete.")

# Generate predictions
with torch.no_grad():
    spikes = torch.zeros((1, num_hidden))
    mem = torch.zeros_like(spikes)
    generated = []
    spike_record = []
    for _ in range(time_steps):
        spikes, mem, output = model(spikes, mem, bias_value=bias_value)
        generated.append(output.squeeze(0))
        spike_record.append(spikes.squeeze(0))
    generated = torch.stack(generated)
    spike_record = torch.stack(spike_record)

# Plot true vs generated sine wave
plt.figure(figsize=(12, 4))
plt.plot(data[:time_steps, 0], label="True Sine Wave")
plt.plot(generated[:, 0], label="Generated Sine Wave")
plt.legend()
plt.title("SNN with FORCE Learning")
plt.show()

# Raster plot
fig, ax = plt.subplots(figsize=(20, 10))
splt.raster(spike_record, ax, s=2, c="black")
plt.title("Raster Plot of Spikes")
plt.xlabel("Time step")
plt.ylabel("Neuron index")
plt.show()
