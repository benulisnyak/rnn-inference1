import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

np.random.seed(seed=0)

zx = average_activity_interp
N = 500  #Number of neurons
dt = 5e-5
tref = 2e-3 #Refractory time constant in seconds
tm = 1e-2 #Membrane time constant
vreset = -65 #Voltage reset
vthr = -40 #Voltage threshold
vpeak = 30

td = 1e-1 #usually 2e-2
tr = 5e-3 #usually 2e-3

alpha = dt*0.1 #Sets the rate of weight change, too fast is unstable, too slow is bad as well.
Pinv = np.eye(N)*alpha #initialize the correlation weight matrix for RLMS
p = 0.1 #Set the network sparsity


nt = int(len(zx))
T = nt*dt
imin = round(0.2*nt) # beginning time step of RLS training
icrit = round(0.5*nt) # end time step of RLS training
step = 50 # weights update time step

Q = 3; G = 0.02;


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


#Simulation
for i in tqdm(range(nt)):
    I = IPSC + E @ z + BIAS #Neuronal Current
    dv = ((dt*i) > (tlast + tref))*(-v + I) / tm #Voltage equation with refractory period
    v = v + dt*dv
    index = np.where(v>=vthr)[0] #Find the neurons that have spiked

    # Store spike times, and get the weight matrix column sum of spikers
    len_idx = len(index)
    if len_idx>0:
        JD = np.sum(OMEGA[:, index], axis=1, keepdims=True) #compute the increase in current due to spiking
        ns = ns + len_idx # total spikes
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
    plt.plot(np.arange(nt)*dt, current[:, j], label=f"Decoded {labels[j]}")
    plt.plot(np.arange(nt)*dt, zx, alpha = 0.5, label=f"Target {labels[j]}")
    plt.axvline(x=imin*dt, color='red', label = 'Training start/end')
    plt.axvline(x=icrit*dt, color='red')
    plt.title('Decoded Output vs Target')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    plt.legend(loc = 'upper right')
    plt.show()
