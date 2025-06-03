import torch
import numpy as np
import matplotlib.pyplot as plt
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import surrogate
import torch.nn as nn
from scipy.integrate import solve_ivp

torch.manual_seed(0)

#Rössler system parameters
a, b, c = 0.2, 0.2, 5.7

def rossler(t, state):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]

#Generate Rössler system data
t_eval = np.linspace(0, 100, 2000)
initial_state = [1.0, 0.0, 0.0]
solution = solve_ivp(rossler, (t_eval[0], t_eval[-1]), initial_state, t_eval=t_eval)

xyz_traj = solution.y.T  #Shape [T, 3]
data = torch.tensor(xyz_traj, dtype=torch.float32)  # [T, 3]

#Network parameters
num_hidden = 2000
num_outputs = 3  # x, y, z
beta = 0.88

class SelfEvolvingSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias_layer = nn.Linear(1, num_hidden, bias=False)
        self.recurrent = nn.Linear(num_hidden, num_hidden, bias=False)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.readout_weights = torch.randn(num_hidden, num_outputs)

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
        out = torch.matmul(spikes, self.readout_weights)
        return spikes, mem, out


model = SelfEvolvingSNN()
alpha = 1.0
P = (1.0 / alpha) * torch.eye(num_hidden)

#training
epochs = 1
time_steps = len(data)
bias_value = 0.5

for epoch in range(epochs):
    spikes = torch.zeros((1, num_hidden))
    mem = torch.zeros_like(spikes)
    for t in range(time_steps):
        target = data[t].unsqueeze(0)  # [1, 3]
        spikes, mem, output = model(spikes, mem, bias_value=bias_value)

        r = spikes.squeeze(0).unsqueeze(1)  # [hidden, 1]
        k = torch.matmul(P, r)
        rPr = torch.matmul(r.T, k)
        c_gain = 1.0 / (1.0 + rPr)
        P -= c_gain * torch.matmul(k, k.T)

        e = output - target  # [1, 3]
        for i in range(num_outputs):
            dw = -e[0, i].item() * c_gain * k
            model.readout_weights[:, i:i+1] += dw

    print(f"Epoch {epoch+1}/{epochs} complete.")

#predicted data
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

# create plots of true vs generated Rössler components
plt.figure(figsize=(14, 6))
labels = ["x", "y", "z"]
for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(data[:, i], label=f"True {labels[i]}")
    plt.plot(generated[:, i], label=f"Generated {labels[i]}")
    plt.legend()
    plt.title(f"{labels[i]} Component")
plt.tight_layout()
plt.show()
