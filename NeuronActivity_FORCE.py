import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

np.random.seed(seed=0)
#zx = average_activity
zx = average_activity_interp
N = 100  #Number of neurons
dt = 5e-4 #normally 5e-5
tref = 2e-3 #Refractory time constant in seconds
tm = 1e-2 #Membrane time constant
vreset = -65 #Voltage reset
vthr = -40 #Voltage threshold
vpeak = 30

td = 1e-0 #usually 2e-2
tr = 1e-3 #usually 2e-3

alpha = dt*0.1 #Sets the rate of weight change, too fast is unstable, too slow is bad as well.
Pinv = np.eye(N)*alpha #initialize the correlation weight matrix for RLMS
p = 0.1 #Set the network sparsity


#Target Dynamics for Product of Sine Waves
#T = 15 # Simulation time (s)

nt = int(len(zx))
T = nt*dt
imin = round(0.1*nt) # beginning time step of RLS training
icrit = round(0.5*nt) # end time step of RLS training
step = 50 # weights update time step
#nt = round(T/dt) # Simulation time step
Q = 1; G = 0.04;
#zx = np.sin(2*math.pi*np.arange(nt)*dt*5) # Target signal

# Track all spikes during the entire run
all_spikes = []  # List of (time, neuron_id) tuples

# Define plotting periods (in steps)
spike_plot_periods = [
    (imin, imin + 2000)
]


k = 1 # number of output unit
IPSC = np.zeros((N,1)) #post synaptic current storage variable
h = np.zeros((N,1)) #Storage variable for filtered firing rates
r = np.zeros((N,1)) #second storage variable for filtered rates
hr = np.zeros((N,1)) #Third variable for filtered rates
JD = np.zeros((N,1)) #storage variable required for each spike time
tspike = np.zeros((4*nt,2)) #Storage variable for spike times
ns = 0 #Number of spikes, counts during simulation
z = np.zeros((k, 1)) #Initialize the approximant
err_list = []
errt_list = []
training_list = []
trainingt_list = []
v = vreset + np.random.rand(N,1)*(vpeak-vreset) #Initialize neuronal voltage with random distribtuions
OMEGA = G*(np.random.randn(N,N))*(np.random.rand(N,N)<p)/(math.sqrt(N)*p) #The initial weight matrix with fixed random weights
BPhi = np.zeros((N, k)) #The initial matrix that will be learned by FORCE method

#Set the row average weight to be zero, explicitly.
for i in range(N):
    QS = np.where(np.abs(OMEGA[i,:])>0)[0]
    OMEGA[i,QS] = OMEGA[i,QS] - np.sum(OMEGA[i,QS], axis=0)/len(QS)

E = (2*np.random.rand(N, k)-1)*Q #n

# arrays to save
current = np.zeros((nt, k)) #storage variable for output current/approximant
tlast = np.zeros((N,1)) #This vector is used to set  the refractory times
BIAS = vthr #Set the BIAS current, can help decrease/increase firing rates.  0 is fine.
decoder_indices_to_track = np.arange(min(10, N))  # First 10 or fewer if N < 10
bphi_history = []  # Will store arrays of shape (10, k)
#bphi_time = []
#Simulation
for i in tqdm(range(nt)):
    I = IPSC + E @ z + BIAS #Neuronal Current
    dv = ((dt*i) > (tlast + tref))*(-v + I) / tm #Voltage equation with refractory period
    v = v + dt*dv
    index = np.where(v>=vthr)[0] #Find the neurons that have spiked

    # Store spike times, and get the weight matrix column sum of spikers
    len_idx = len(index)
    if len_idx>0:
        for neuron_id in index:
            all_spikes.append((i*dt, neuron_id))
        JD = np.sum(OMEGA[:, index], axis=1, keepdims=True)
        ns = ns + len_idx
        #JD = np.sum(OMEGA[:, index], axis=1, keepdims=True) #compute the increase in current due to spiking
        #ns = ns + len_idx # total spikes
    tlast = tlast + (dt*i - tlast)*(v>=vthr) #Used to set the refractory period of LIF neurons
    # synapse for double exponential
    IPSC = IPSC*math.exp(-dt/tr) + h*dt
    h = h*math.exp(-dt/td) + JD*(len_idx>0)/(tr*td) #Integrate the current
    r = r*math.exp(-dt/tr) + hr*dt
    hr = hr*math.exp(-dt/td) + (v>=vthr)/(tr*td)

    #Implement RLMS with FORCE
    z = BPhi.T @ r #approximant
    err = z - zx[i:i+1].T
    if i > imin:
      if i < icrit:
        err_list.append(err[0])
        errt_list.append(i)
    # RLMS
    if i % step == 1:
        if i > imin:
            if i < icrit:
                cd = (Pinv @ r)
                BPhi = BPhi - (cd @ err.T)
                Pinv = Pinv - (cd @ cd.T) / (1.0 + r.T @ cd)
                training_list.append(err[0])
                trainingt_list.append(i)
    v = v + (vpeak - v)*(v>=vthr) # set peak voltage
    v = v + (vreset - v)*(v>=vthr) #reset with spike time interpolant implemented.
    current[i, :] = z[:, 0]
    bphi_history.append(BPhi[decoder_indices_to_track, 0].copy())

plt.figure(figsize = (15,6))
plt.plot(errt_list, err_list, 'o', markersize = 2)
plt.plot(trainingt_list, training_list, 'o', markersize = 2, label = 'Training step', color = 'red')
plt.legend(loc = 'lower right')
plt.xlabel('Time step')
plt.ylabel('Error')
plt.title('Error during training')
plt.show()

#creates plots
labels = ['x(t)', 'y(t)', 'z(t)'] #assuming 3 dimensions, change if you change zx
for j in range(k):
    plt.figure(figsize=(14, 6))
    plt.plot(np.arange(nt), current[:, j], label=f"Decoded {labels[j]}")
    plt.plot(np.arange(nt), zx, alpha = 0.5, label=f"Target {labels[j]}")
    plt.axvline(x=imin, color='red', label = 'Training start/end')
    plt.axvline(x=icrit, color='red')
    plt.title('Decoded Output vs Target')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.legend(loc = 'upper right')
    plt.show()

all_spikes = np.array(all_spikes)
all_times = all_spikes[:, 0]
all_neurons = all_spikes[:, 1].astype(int)

for period_idx, (start_step, end_step) in enumerate(spike_plot_periods):
    start_time = start_step * dt
    end_time = end_step * dt

    # Filter spikes in time window
    mask = (all_times >= start_time) & (all_times <= end_time)
    times = all_times[mask]
    neurons = all_neurons[mask]

    if len(times) > 0:
        plt.figure(figsize=(14, 6))
        plt.eventplot(
            positions=[times[neurons == n] for n in range(N) if np.any(neurons == n)],
            lineoffsets=[n for n in range(N) if np.any(neurons == n)],
            linelengths=0.8,
            colors='black'
        )
        plt.xlabel('Time (s)')
        plt.ylabel('Neuron index')
        plt.title(f'Raster Plot for Period {period_idx+1}: {start_time:.2f}s to {end_time:.2f}s')
        plt.ylim(-1, N)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No spikes recorded during period {period_idx+1}.")

bphi_history = np.array(bphi_history)  # Shape: (time_steps, 10)

# Plot evolution of decoder weights
plt.figure(figsize=(14, 6))
for i in range(bphi_history.shape[1]):
    plt.plot(np.arange(nt)*dt, bphi_history[:, i], label=f'BPhi[{i}, 0]')
plt.xlabel('Time (s)')
plt.ylabel('Decoder Weight Value')
plt.title('Evolution of First 10 Decoder Weights in BPhi')
#plt.legend(loc='upper right', ncol=2)
#plt.grid(True)
#plt.tight_layout()
plt.show()
