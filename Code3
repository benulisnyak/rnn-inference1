import torch
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import surrogate
import torch.nn as nn
import matplotlib.pyplot as plt
from snntorch import spikeplot as splt

torch.manual_seed(0)

def rossler(t, state, a=0.2, b=0.2, c=5.7):
  x, y, z = state
  dxdt = -y - z
  dydt = x + a*y
  dzdt = b + z*(x - c)
  return [dxdt, dydt, dzdt]

#Solve Rössler system
t_span = (0, 500)
t_eval = np.linspace(*t_span, 10000)
sol = solve_ivp(rossler, t_span, [1, 1, 1], t_eval=t_eval)

#Normalize the data
data = torch.tensor(sol.y, dtype=torch.float32).T
data = (data - data.mean(0)) / data.std(0)

#Network stuff:
num_hidden = 500
num_outputs = 3
beta = 0.88 #decay parameter - larger values mean faster firing rate


#Define Network
class SelfEvolvingSNN(nn.Module):
  def __init__(self):
    super().__init__()
    #self.fc1 = nn.Linear(num_outputs, num_hidden) #.fc1 applies linear transform to the input data
    self.recurrent = nn.Linear(num_hidden, num_hidden)
    self.lif1 = snn.Leaky(beta = beta, spike_grad=surrogate.fast_sigmoid())
    self.readout = nn.Linear(num_hidden, 3)

  def forward(self, spikes, mem2):
    #mem1 = self.lif1.init_leaky(batch_size=1, shape = (num_hidden)) + mem2
    bias = 0.15
    h = self.recurrent(spikes) * 0.25 + bias
    spk1, mem2 = self.lif1(h, mem2)
    out = self.readout(spk1)
    return spk1, mem2, out


######## Training
model = SelfEvolvingSNN()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

#Initial state
init_state = data[0]  # [X0, Y0, Z0]
# Training loop
epochs = 100 #Number of backpropogations loops
time_steps = 2000 #shorter for training
num_inputs = 3


for epoch in range(epochs):
  spikes = torch.zeros((1, num_hidden))
  mem2 = torch.zeros_like(spikes)
  epoch_loss = 0.0
  for t in range(time_steps):
    optimizer.zero_grad()
    spikes, mem2, output = model(spikes, mem2)
    target = data[t]
    loss = loss_fn(output.squeeze(), target)
    loss.backward()
    optimizer.step()
    spikes = spikes.detach()
    mem2 = mem2.detach()
    epoch_loss += loss.item()  #Add current loss

  avg_loss = epoch_loss / time_steps
  print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.6f}")

with torch.no_grad():
    spikes = torch.zeros((1, num_hidden))
    mem2 = torch.zeros_like(spikes)
    generated = []
    spike_record = []
    for _ in range(time_steps):
        spikes, mem2, output = model(spikes, mem2)
        generated.append(output.squeeze(0))
        spike_record.append(spikes.squeeze(0))
    generated = torch.stack(generated)
    spike_record = torch.stack(spike_record)


#Creates plots
plt.figure(figsize=(12, 4))
plt.plot(data[:time_steps, 0], label="True X")
plt.plot(generated[:, 0], label="Generated X", linestyle='--')
plt.legend()
plt.title("Comparison of X Dynamics")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(data[:time_steps, 1], label="True Y")
plt.plot(generated[:, 1], label="Generated Y", linestyle='--')
plt.legend()
plt.title("Comparison of Y Dynamics")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(data[:time_steps, 2], label="True Z")
plt.plot(generated[:, 2], label="Generated Z", linestyle='--')
plt.legend()
plt.title("Comparison of Z Dynamics")
plt.show()


fig, ax = plt.subplots(figsize=(20, 10))
splt.raster(spike_record, ax, s=2, c="black")  # Spike matrix: [time, neurons]
plt.title("Raster Plot of Spikes")
plt.xlabel("Time step")
plt.ylabel("Neuron index")
plt.show()
