from __future__ import division
import tkinter as tk
from tkinter import ttk
import numpy as np
import random
import matplotlib.pyplot as plt
from numba import jit
import time as runtime
from scipy import signal
from collections import deque
import networkx as nx
import scipy.spatial.distance as spd
from configparser import SafeConfigParser
from collections import Counter
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length
from networkx.algorithms.cluster import average_clustering
import Auxillary_funcs as AF
import matplotlib.colors as col
import matplotlib.cm as cm

path = 'C:\\Users\\catha\\Documents\\Research stuff\\Masters thesis\\'

def register_own_cmaps():
    
    startcolor = '#FFFFFF' 
    color1 = '#F2F2F2'   
    color2 = '#E1D8F5'    
    color3 = '#D1B6F5' 
    endcolor = '#442273'    
    
    cmap2 = col.LinearSegmentedColormap.from_list('STN_Raster',[startcolor,color2,endcolor,endcolor]) # color1,color2,color3,
    # extra arguments are N=256, gamma=1.0
    cm.register_cmap(cmap=cmap2)
    
    
    startcolor = '#FFFFFF'
    color1 = '#F2F2F2' 
    color2 = '#96C2A7'    
    color3 = '#6CC28A' 
    endcolor = '#038C33'    
    
    cmap2 = col.LinearSegmentedColormap.from_list('GPe_Raster',[startcolor,color2,endcolor,endcolor]) #,color1,color2,color3,
    # extra arguments are N=256, gamma=1.0
    cm.register_cmap(cmap=cmap2)

register_own_cmaps()

@jit(nopython=True,nogil=True) 
def cortical_synapse(ip,h,samp,no):
    
    op = np.zeros((samp,no))
    op_aux = np.zeros((samp,no))

    for i in range(samp):
        for j in range(no):
            op[i,j] = op[i-1,j] + h*op_aux[i-1,j] 
            op_aux[i,j] = op_aux[i-1,j] + (np.e*15)*ip[i,j] - h*(2*op_aux[i-1,j] +op[i-1,j])
 
    return op

class alpha_synapse:
    def __init__ (self):
        #Synapse parameters
        #tonic spiking A = 14, tonic bursting A = 15
        self.A = 15     #magnitude
        self.tc = 0.8    #time constant
        
        #Syanapse output (units are mV I think, or maybe its current)
        self.op = 0
        self.op_aux = 0
        self.ip = 0
        
    def update(self,h=0.01):
        self.op = self.op + h*self.op_aux
        self.op_aux = self.op_aux + (np.e*self.A/self.tc)*self.ip - h*(2*(1/self.tc)*self.op_aux +((1/self.tc)**2)*self.op)
        self.ip = 0
        
    def spike(self,thresh=1):
        if self.op<=thresh:
            self.ip = 1

#It seems to be always faster to just use the default python version of the last sig function
#Doing it in a loop with jit gives the same performance.
def fast_sig(x):
    a = x/(1+abs(x))
    return a
 
@jit(nopython=True,nogil=True)  
def update(inter,udec,h,V,U,op,op_aux,no,param):
    a = param[0]
    b = param[1] 
    c = param[2]     
    d = param[3]
    
    ip = np.zeros(no)

    #within the function your creating a new variable and assiging it V = ...
    #if you want to update the original variable use V[:] = ...
    #Does this run faster if you replace the indexing with loops?
    V[:] = V + h*(0.04*V**2 + 5*V +140 - U + inter)
    U[:] = U + h*(a*(b*V-U) + udec)   

    for j in range(no):
        if V[j]>9 and V[j]<30:
            V[j] = 30
        elif V[j]>=30:
            V[j] = c[j]
            U[j] = U[j] + d[j]
            ip[j] = 1

    op[:] = op + h*op_aux 
    op_aux[:] = op_aux + (np.e*15)*ip - h*(2*(1)*op_aux +op)
    
    
def update_GPe(I,h,GPe):
    #Unpack the dictionary
    V = GPe["V"]
    U = GPe["U"]
    op = GPe["op"]
    op_aux = GPe["op_aux"]
    E = GPe["E"]
    Lock = GPe["Lock"]
    offset = GPe["offset"]
    no = np.size(V,axis = 0)
    
    inter = np.zeros(no)
    #in this function it makes more sense to add the offset here, because there are multiple different offsets
    if np.array([I>=-2.2]).any() == True:
        inter[I>=-2.2] = 30*fast_sig(I[I>=-2.2].real/30) +offset #+ ran_offset
        
    if np.array([I<-2.2]).any() == True:
        inter[I<-2.2] =  -7.5*fast_sig(I[I<-2.2].real/1)
    
    udec = 0 
    
    #Could use a threshold instead of imaginary number
    Lock[(I.imag>0) | (I.imag<0)] = int(5/h)
    
    inter[Lock>0] = -5
    #udec[Lock>0] = 0
    U[Lock>0] = 0
    
    Lock[Lock>0] = Lock[Lock>0]  - 1

    #Parameters
    a = 0.02*np.ones(no,dtype='float') #+min(0.08,B) #this ensures the plateau potential happens faster
    b = 0.2*np.ones(no,dtype='float') 
    c = -65*np.ones(no,dtype='float')    
    d = 6*np.ones(no,dtype='float')    
    
    a[I<-2.2] = 0.0002
    b[I<-2.2] = 0.195
    c[I<-2.2] = -60
    d[I<-2.2] = 0.06
    
    #if its less than -8, I want to move it towards -8
    #U = U - 0.1(U--8)
    #U[(I<-2.2) & (abs(U[I<-2.2])>4)] = U[I<-2.2] - 0.01*(U[I<-2.2] +9.5)
    #aux = U[abs(U[I<-2.2]>4)]
    #aux = U[ (I<-2.2)& (U>8)]
    
    threshold = -8.75
    U[ (I<-2.2)& (U>threshold)] = U[ (I<-2.2)& (U>threshold)] - 0.01*h*(U[ (I<-2.2)& (U>threshold)] -threshold)
    
    param = np.array([a,b,c,d],dtype='float')

    #update(inter+np.random.uniform(-1,1,no),udec,h,V,U,op,op_aux,no,param)
    update(inter,udec,h,V,U,op,op_aux,no,param)
    
    
    GPe["V"] = V
    GPe["U"] = U
    GPe["op"] = op
    GPe["op_aux"] = op_aux
    GPe["E"] = E
    GPe["Lock"] = Lock

def update_STN(I,h,STN):
    #Unpack the dictionary
    V = STN["V"]
    U = STN["U"]
    op = STN["op"]
    op_aux = STN["op_aux"]
    B = STN["B"]
    plateau_count = STN["plateau_count"]
    plateau_dur = STN["plateau_dur"]
    offset = STN["offset"]
    Plateau = STN["Plateau"]
    no = np.size(V,axis = 0)

    inter = np.zeros(no)
    inter[I>=0] = 50*fast_sig(I[I>=0]/100)
    
    udec = np.zeros(no)
    udec[I<0] = -2*fast_sig(I[I<0]/3)
    
    #Parameters
    a = 0.02*np.ones(no,dtype='float')
    a[B>0] = 0.1  #this ensures the plateau potential happens faster
    b = -0.01*np.ones(no,dtype='float') 
    c = -60*np.ones(no,dtype='float')    
    d = 4*np.ones(no,dtype='float')    
    param = np.array([a,b,c,d],dtype='float')
    
    
    #Using index fuckery to avoid loops and make the program far more efficient
    #This bit is for starting the plateau potential
    plateau_count[(B>0)&(V>-30)] = plateau_dur + np.random.randint(-(int(5/h)),(int(5/h)),size =np.size(plateau_count[(B>0)&(V>-30)]) )
    
    B[(B>0)&(V>-30)] = False

    #Update
    inter[plateau_count>0] = 28
    udec[plateau_count>0] = 0
    
    #This allows U to be modified when the plateau ends
    pre = np.zeros(no,dtype='bool')
    
    pre[plateau_count>0] = 1

    update(inter+offset,udec,h,V,U,op,op_aux,no,param)
    
    #Commenting out this adaptive plateau potential bit because it does not work the way it was intended
    #Increase plateau duration with sustained hyperpolarising input
    #plateau_count[(plateau_count>0)&(I<-20)] = plateau_count[(plateau_count>0)&(I<-20)] + 5
    #Cut plateau duration if there is a strong positive input
   # plateau_count[(plateau_count>0)&(I<-20)&(plateau_count<int(3*plateau_dur/4))] = plateau_count[(plateau_count>0)&(I<-20)&(plateau_count<int(3*plateau_dur/4))] - plateau_count[(plateau_count>0)&(I<-20)&(plateau_count<int(3*plateau_dur/4))] *h/4
    # 
    # index = (plateau_count>0)&(I<-20)&(plateau_count<int(3*plateau_dur/4))
    # if plateau_count[index ].any() == True:
    #     print(np.sum(plateau_count[index ]))
    # plateau_count[index ] = plateau_count[index] - plateau_count[index] #plateau_count[index]/4000
    
    #Always decrease plateau count
    plateau_count[plateau_count>0] = plateau_count[plateau_count>0] - 1

    U[(pre)&(plateau_count==0)] = U[(pre)&(plateau_count==0)] -20
    
    B[(V<-84)&(Plateau == True)] = 1

    #Repack the list
    STN["V"] = V
    STN["U"] = U
    STN["op"] = op
    STN["op_aux"] = op_aux
    STN["B"] = B 
    STN["plateau_count"] = plateau_count
    
    
@jit(nopython=True,nogil=True)
def Input_from_connections(Connection_Matrix,Input,no):
    output = np.zeros(no)
    for i in range(no):
        p = 0
        for j in range(no):
            p = Connection_Matrix[i,j]*Input[i]
        output[i] = p
    
    return output
    
@jit(nopython=True,nogil=True) 
def connect_a_list(Output,connections,no):
    #This is a surpisingly robust way to calculate the input from a list of neuron
    #with no constraints other than that the recieving (i.e the [1]) part of the list is ordered
    # i is the neuron its going to, j, is the neuron its coming from
    j=0
    Input = np.zeros(no)
    for i in connections[:,1]:
        # sum the weighted neuron inputs, edge_list[j,2] is the weight
        Input[i] = Input[i] +Output[connections[j,0]]
        j = j+1
        
    return Input
 
# @jit(nopython=True,nogil=True)   
# def connect_a_list_weights(Output,connections,no,weights):
#     #This is a surpisingly robust way to calculate the input from a list of neuron
#     #with no constraints other than that the recieving (i.e the [1]) part of the list is ordered
#     # i is the neuron its going to, j, is the neuron its coming from
#     j=0
#     Input = np.zeros(no)
#     for i in connections[:,1]:
#         # sum the weighted neuron inputs, edge_list[j,2] is the weight
#         Input[i] = Input[i] + weights[j]*Output[connections[j,0]]
#         j = j+1
#         
#     return Input


def Cortico_STN_inputs(no_STN,F,h,t,degree=0.5,phase = 0):
    offset = 100
    deltaT = 1000*phase/(F*np.pi)
    s = int(round(deltaT/h))
    Ts = int(1000/(F*h))
    inputs = np.zeros( (int((t+offset)/h),no_STN), dtype = bool )
    cortex = np.zeros( (int((t+offset)/h),no_STN),dtype = bool )
    
    no_synchronised = int(round(degree*no_STN))
    
    for i in range(int((t+offset)/(h*Ts))):
        inputs[Ts*i+s,0:no_synchronised] = 1
    
    inputs[:,no_synchronised:no_STN] = np.random.randint(0, Ts,size = (int((t+offset)/h),no_STN - no_synchronised) ) == 0
    
    np.random.shuffle(np.transpose(inputs))

    cortex = cortical_synapse(inputs,h,int((t+offset)/h),no_STN)

    #np.random.shuffle(np.transpose(cortex))

    ind1 = int(offset/h)
    ind2 = int((t+offset)/h)
    cortex = cortex[ind1:ind2,:]
    
    return cortex

def Striatal_GPe_inputs(no_GPe,F,h,t,degree=0.5,phase = 0):
    offset = 100
    deltaT = 1000*phase/(F*np.pi)
    s = int(round(deltaT/h))
    Ts = int(1000/(F*h))
    
    cortex = np.zeros( ( (int( (t+offset)/h)) , no_GPe ) , dtype = 'complex' )
    
    no_synchronised = int(round(degree*no_GPe))
 
    random_numbers = np.random.randint(0, Ts,size = (int((t+offset)/h),no_GPe) ) == 0
    
    cortex[random_numbers] = 1j
    cortex[:,0:no_synchronised] = False
    
    for i in range(int((t+offset)/(h*Ts))):
        cortex[Ts*i+s,0:no_synchronised] = 1j
        
    ind1 = int(offset/h)
    ind2 = int((t+offset)/h)
    cortex = cortex[ind1:ind2,:]
    
    np.random.shuffle(np.transpose(cortex))
    
    return cortex

def Raster_plot(act,step=1,cmap= 'STN_Raster',bottom=True, label = 'STN'):
    time = np.size(act[0,:])
    length = np.size(act[:,0])
    sc = int(time/(2*length*step))
    
    act2 = np.zeros((sc*length,int(time/step)))
    
    try:
        act2[0,:] = act[0,0:time:step]
        act2 = np.zeros((sc*length,int(time/step)))
    except:
        act2 = np.zeros((sc*length,int(time/step)+1))
        
    for i in range(length):
        for j in range(int(sc/2)):
            act2[sc*i + j:] = act[i,0:time:step]
            
       
    plt.imshow(act2,origin='lower',interpolation = 'none',aspect = 'auto', extent = [0,time*0.25,0,length], cmap = cmap )
    plt.title(label + ' Raster Plot')
    plt.ylabel('Neurons')
    
    
    if bottom:
        plt.xlabel('Time (ms)')
        
    if not bottom:
        plt.xticks([])
        
    #plt.grid()
        
    
    plt.show()
    
    #return act2

def plot_results(t,STN,GPe,plots):
    B = np.mean(STN,axis=0)
    B2 = np.mean(GPe,axis=0)
    STN_shape = np.shape(STN)
    GPe_shape = np.shape(GPe)
    
    
    if plots[0]:
        plt.figure(0)
        plt.subplot(2,1,1)
        Raster_plot(STN,step=1,cmap = 'STN_Raster',bottom = False,label = 'STN')
        plt.subplot(2,1,2)
        Raster_plot(GPe,step=1,cmap = 'GPe_Raster',label = 'GPe')
        #plt.show()
    
    if plots[1]:
        fig, (ax1,ax2) = plt.subplots(2,1, sharex = True,sharey = True)
        
        fig.set_figheight(7,forward=True)
        fig.set_figwidth(8,forward=True)
        
        fig.suptitle('Local Field Potential')
        
        ######################
        #STN
        ax1.plot(t,B,'#442273')
        ax1.set(ylabel = 'Mean Synaptic Output (A.U)')
        
        mu = np.mean(B[1000:STN_shape[1]])
        pop_var = np.var(B[1000:STN_shape[1]]-mu)
        neur_var = np.var(STN[:,1000:STN_shape[1]],axis=1)
        synchrony = pop_var/neur_var.mean()
        
        
        EV_STN = 1/(np.sqrt(STN_shape[0]))
        
        textstr = '\n'.join( (
            r'$\mu=%.3f$' % (mu, ),
            r'$\chi^2=%.3f$'% (synchrony,)))#,
            #r'$E(\chi^2)=%.3f$'% (EV_STN,) ))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        # ax1.text(0.855, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        #         verticalalignment='top', bbox=props)
        ax1.legend(['STN'],loc = 1,fontsize = 12 )
        
        ######################
        #GPe
        ax2.plot(t,B2,color='#038C33')
        ax2.set(xlabel = 'Time (ms)', ylabel = 'Mean Synaptic Output (A.U)')
        
        mu = np.mean(B2[1000:GPe_shape[1]])
        pop_var = np.var(B2[1000:GPe_shape[1]]-mu)
        neur_var = np.var(GPe[:,1000:GPe_shape[1]],axis=1)
        synchrony = pop_var/neur_var.mean()
        
        
        EV_GPe = 1/(np.sqrt(GPe_shape[0]))
        
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (mu, ),
            #r'$\mathrm{median}=%.2f$' % (median, ),
            r'$\chi^2=%.3f$'% (synchrony, ) ))#,
            #r'$E(\chi^2)=%.3f$'% (EV_GPe,) ))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        # ax2.text(0.855, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
        #         verticalalignment='top', bbox=props)
        ax2.legend(['GPe'],loc = 1,fontsize = 12 )
        
        
        
    if plots[2]:
        
        nps = 4096
        f,STN_freq = signal.welch(B[4000:(np.size(t))], 1000/0.25, nperseg=nps,noverlap= int(nps/2))
    
        a = int((80/4000)*nps)
        
        fig, (ax1,ax2) = plt.subplots(2,1, sharex = True,sharey = True)
        fig.set_figheight(7,forward=True)
        fig.set_figwidth(6,forward=True)
        fig.suptitle('Power Spectra')
        
        ax1.plot(f[0:a],STN_freq[0:a],'#442273')
        ax1.set(ylabel = 'Power (A.U.)')
        ax1.legend(['STN'],loc = 1,fontsize = 12 )
        
        f,GPe_freq = signal.welch(B2[4000:(np.size(t))], 1000/0.25, nperseg=nps,noverlap = int(nps/2))

        ax2.plot(f[0:a],GPe_freq[0:a],color='#038C33')
        ax2.set(xlabel='Frequence (Hz)',ylabel = 'Power (A.U.)')
        ax2.legend(['GPe'],loc = 1,fontsize = 12 )
        
    if plots[3]:
        (f,Cxy)= signal.coherence(B[800:(np.size(t))],B2[800:(np.size(t))],fs=4000)
        plt.figure(3)
        plt.plot(f,Cxy)
        plt.show()
        plt.title('Magnitude squared coherence between LFPs of STN and GPe')
        plt.xlabel('Frequence (Hz)')
    
    plt.show()


def STN_GPe(Cortex_gain=2,GPe_gain=0.1,STN_gain=0.5,InterGPe_gain=0.5,
Striatal_gain=1,time =1000,delay = (6/0.25),STN_DC_offset = 0,GPe_DC_offset = 0,
DBS = False, cortex_frequency = 20, cortex_correlated = False, cortex_phase = 0,
striatum_frequency = 3, striatum_phase = 0, striatum_correlated = False, plots=(True,True,True,False)):

    Cortex_gain = float(Cortex_gain)
    GPe_gain = float(GPe_gain)
    STN_gain = float(STN_gain)
    InterGPe_gain = float(InterGPe_gain)
    Striatal_gain = float(Striatal_gain)
    time = int(time)
    delay = int(delay)
    STN_DC_offset = float(STN_DC_offset)
    GPe_DC_offset = float(GPe_DC_offset)
    DBS = False 
    cortex_frequency = float(cortex_frequency) 
    cortex_correlated = float(cortex_correlated)
    cortex_phase = float(cortex_phase)
    striatum_frequency = float(striatum_frequency) 
    striatum_correlated = float(striatum_correlated)
    striatum_phase = float(striatum_phase)
    

    t0 = runtime.time()
    

    Cortex_gain = float(Cortex_gain)
    GPe_gain = float(GPe_gain)
    STN_gain = float(STN_gain)
    InterGPe_gain = float(InterGPe_gain)
    Striatal_gain  = float(Striatal_gain)
    
    h = 0.25
    delay = int(delay)

    GPetoGPe = np.load(path+'Intermediate Data\\GPe Interconnections.npy')
    STNtoGPe = np.load(path+'Intermediate Data\\fromSTNtoGpe.npy')
    STNfromGPe = np.load(path+'Intermediate Data\\fromGPetoSTN.npy')

    no_STN = int(max(STNfromGPe[1,:]) + 1)
    no_GPe = int(max(STNtoGPe[1,:]) + 1)
    
    Cortex = Cortico_STN_inputs(int(no_STN),cortex_frequency,h,int(time),cortex_correlated,cortex_phase)
    
    if DBS:
        DBS = 10*Cortico_STN_inputs(int(no_STN),140,h,int(time),True)
        Cortex = Cortex + DBS
        
    Striatum = Striatal_GPe_inputs(no_GPe,striatum_frequency,h,time,striatum_correlated,striatum_phase) #the striatum is largely quiscient
    
    STN_DC_offset = STN_DC_offset #lower in pd, higher in normal conditions
    GPe_DC_offset = GPe_DC_offset #-2.2
    #need to balance excitation and inhibition to some extent

    
    print('No. STN: ',no_STN)
    print('No. GPe: ',no_GPe)

    GPe_input = deque()
    for i in range(delay):
        GPe_input.append(np.zeros(no_GPe))
        
    STN_input = deque()
    for i in range(delay):
        STN_input.append(np.zeros(no_STN))

    
    STN = {"V": -65*np.ones(no_STN),
            "U": 8*np.ones(no_STN), #np.random.uniform(8,14,no_STN), #12*np.ones(no_STN),
            "op": np.zeros(no_STN),
            "op_aux": np.zeros(no_STN),
            "B": np.zeros(no_STN,dtype = 'bool'),
            "plateau_count":np.zeros(no_STN,dtype = 'int'),
            "plateau_dur":int(180/h), #literature says 200, but 40 would allow beta
            "offset": np.random.uniform(16.9,18.2,no_STN), #16.95*np.ones(no_STN) + np.random.uniform(-0.005,0.005,no_STN)
            "Plateau": np.random.randint(0,2,no_STN)
            }
    GPe = {"V": -65*np.ones(no_GPe),
            "U": np.random.uniform(-12,-7,no_GPe),#-9*np.ones(no_GPe),
            "op": np.zeros(no_GPe),
            "op_aux": np.zeros(no_GPe),
            "Lock": np.zeros(no_GPe,dtype = 'bool'),
            "E":np.zeros(no_GPe,dtype = 'int'),
            "offset":5.225,
            }

    
    t = np.linspace(0,time,int(time/h))
    
    Total = np.zeros((no_STN,len(t)))
    Total2 = np.zeros((no_GPe,len(t)))
    
    
    for i in range(len(t)):
        GPe_inter = -1*connect_a_list(GPe["op"],np.transpose(GPetoGPe),no_GPe)
        GPe_input.append(connect_a_list(STN["op"],np.transpose(STNtoGPe),no_GPe))
        STN_input.append(-1*connect_a_list(GPe["op"],np.transpose(STNfromGPe),no_STN))
        
        IP_to_GPe = Striatal_gain*Striatum[i,:] + InterGPe_gain*GPe_inter + STN_gain*GPe_input.popleft() + GPe_DC_offset
        
        update_GPe(IP_to_GPe,h,GPe)
        
        IP_to_STN = Cortex_gain*Cortex[i,:] + GPe_gain*STN_input.popleft() + STN_DC_offset
        
        update_STN(IP_to_STN,h,STN)
    
        Total[:,i] = STN["op"]
        Total2[:,i] = GPe["op"]
        

    t1 = runtime.time()
    
    plot_results(t,Total,Total2,plots)
    
    print('Time taken: ',t1-t0)
    
    
#############################################
#Network generation
#############################################

def Connection_distribution(edge_list,no_neur,typ):
    cnt = Counter(range(no_neur))

    for i in range(len(edge_list[0,:])):
        #to see in degree set it to 1
        #to see out degree set it to 0
        cnt[edge_list[typ,i]] +=1
    
    edge_dist = np.zeros((no_neur,2),dtype = int)
    
    for i in range(no_neur):
        edge_dist[i,0] = i
        edge_dist[i,1] = cnt[i]-1
    
    ind = np.argsort(edge_dist[:,1])
    
    sdist = edge_dist[ind]
    sdist = sdist[::-1]
    
    return sdist

    
def calc_SW(GPetostn,STNtogpe,GPe_interconnections,no_GPe,no_STN,k):
    STNtogpe[0,:] += no_GPe
    
    GPetostn[1,:] += no_GPe

    edges = np.concatenate((STNtogpe,GPetostn),axis=1)
    
    
    gpe_inter = GPe_interconnections.astype(int)

    edges = np.concatenate((edges,gpe_inter),axis = 1)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(no_STN+no_GPe)) 
    G.add_edges_from(np.transpose(edges))
    
    (Reference, w)  = C_random_k_in_graph(no_STN+no_GPe, k, 10000*k*(no_STN+no_GPe))
    
    R = nx.DiGraph()
    R.add_nodes_from(range(no_STN+no_GPe)) 
    R.add_edges_from(np.transpose(Reference))
    
    RClust = average_clustering(R)
    RPath = average_shortest_path_length(R)
    
    GClust = average_clustering(G)
    GPath = average_shortest_path_length(G)
    
    
    SW = (GClust/RClust)/(GPath/RPath)
    
    return SW


def Connection_distribution(edge_list,no_neur,typ):
    cnt = Counter(range(no_neur))

    for i in range(len(edge_list[0,:])):
        #to see in degree set it to 1
        #to see out degree set it to 0
        cnt[edge_list[typ,i]] +=1
    
    edge_dist = np.zeros((no_neur,2),dtype = int)
    
    for i in range(no_neur):
        edge_dist[i,0] = i
        edge_dist[i,1] = cnt[i]-1
    
    ind = np.argsort(edge_dist[:,1])
    
    sdist = edge_dist[ind]
    sdist = sdist[::-1]
    
    return sdist

@jit(nopython = True, nogil = True)
def C_weighted_choice(Nodes,Weights):
    """Returns a single element from a weighted sample.

    The input is a dictionary of items with weights as values.
    """
    #use roulette method to choose Node given weight
    rnd = random.random() * np.sum(Weights)

    
    for k in range(len(Nodes)):
        rnd -= Weights[k]
        if rnd <0:
            return Nodes[k]
            
                   
@jit(nopython = True, nogil = True)
def IsInArray(element,array,array_size):
    array_size = int(array_size)
    for i in range(array_size):
        if (element[0] == array[0,i]) & (element[1] == array[1,i]):
            return True
    
    return False
    

    
def SG_helper_func(i,k,GPe_in_degree,GPenodes,STNnodes,STNWeights,edge_list,edge_list_size,err_count,GPe):
    element = np.zeros(2)
    useable_in = np.zeros(GPe)
    #Select the GPe neurons with in degree less than k
    for p in range(GPe):
        if GPe_in_degree[p]<k:
            useable_in[p] = p
        else:
            useable_in[p] = -1

    V = np.random.choice(useable_in[useable_in>=0])
    
    GPe_in_degree[int(V)] +=1

    #v comes from a weighted choice determined by weights
    useable_in = STNnodes.copy()
    useable_weights = STNWeights.copy()

    #if the edge already exists in the list it randomly chooses another U node
    chk = 1
    while(chk != 0):
        U = C_weighted_choice(useable_in[useable_in>=0], useable_weights[useable_weights>=0])
        element[0] = U
        element[1] = V
        if IsInArray(element,edge_list,edge_list_size): #[U,V] in edge_list
            err_count[0] +=1
            chk+=1
        else:
            chk=0
        if chk> 2*GPe: #have to give some limit on number of times to try
            raise ValueError('cannot create network, try lower connectivity')
        
    edge_list[0,i] = U
    edge_list[1,i] = V
    
    STNWeights[int(U)] +=1  


def K_In_Diff_Nets(STN,GPe, k, alpha):
    """Generates one way connections from STN to GPe
    k is the in-degree of GPe neurons
    Prefferential attachment with respect to the STN based on alpha:
        STN neurons more likely to be chosen based on their weight.
        Alpha is the initial weight of all the STN neurons.
    """

    if isinstance(alpha, np.ndarray):
        STNWeights = alpha
    elif alpha < 0:
        raise ValueError('alpha must be positive')
    else:    
        STNWeights = alpha*np.ones(STN)
    
    STNnodes = np.arange(STN)         
    #STNWeights = alpha*np.ones(STN)
    
    GPenodes = np.arange(GPe)
    GPe_in_degree = np.zeros(GPe)
    
    edge_list = np.zeros((2,GPe*k))
    err_count = np.zeros(1)
    
    edge_list_size = 0

    for i in range(k * GPe):
        edge_list_size +=1
        SG_helper_func(i,k,GPe_in_degree,GPenodes,STNnodes,STNWeights,edge_list,edge_list_size,err_count,GPe)
    
    #print('Retries ',err_count)

    return edge_list



@jit(nopython = True, nogil = True)
def helper_func(i,n,k,in_degree,nodes,Weights,edge_list,edge_list_size,err_count):
    element = np.zeros(2)
    useable_in = np.zeros(n)
    #Select the GPe neurons with in degree less than k
    for p in range(n):
        if in_degree[p]<k:
            useable_in[p] = p
        else:
            useable_in[p] = -1

    V = np.random.choice(useable_in[useable_in>=0])
    
    in_degree[int(V)] +=1

    #U comes from a weighted choice determined by weights
    useable_in = nodes.copy()
    useable_weights = Weights.copy()
    useable_in[int(V)] = -1
    useable_weights[int(V)] = -1

    #if the edge already exists in the list it randomly chooses another U node
    chk = 1
    while(chk != 0):
        U = C_weighted_choice(useable_in[useable_in>=0], useable_weights[useable_weights>=0])
        element[0] = U
        element[1] = V
        if IsInArray(element,edge_list,edge_list_size): #[U,V] in edge_list
            err_count[0] +=1
            chk+=1
        else:
            chk=0
        if chk> 2*n: #have to give some limit on number of times to try
            raise ValueError('cannot create network, try lower connectivity')
        
    edge_list[0,i] = U
    edge_list[1,i] = V
    
    Weights[int(U)] +=1  
    

def C_random_k_in_graph(n, k, alpha):    
    if isinstance(alpha, np.ndarray):
        Weights = alpha
    elif alpha < 0:
        raise ValueError('alpha must be positive')
    else:    
        Weights = alpha*np.ones(n)
        
    in_degree = np.zeros(n)
    nodes = np.arange(n)
    edge_list = np.zeros((2,n*k))
    err_count = np.zeros(1)
    
    edge_list_size = 0

    for i in range(k * n ):
        edge_list_size +=1
        helper_func(i,n,k,in_degree,nodes,Weights,edge_list,edge_list_size,err_count)
    
    #print('Retries ',err_count)

    return edge_list
    
    
def show_results(no_STN,no_GPe,GPe_interconnections,STNtogpe,GPetostn,k,disp = True):
    name='K_out'

    STNtogpe[0,:] += no_GPe
    
    GPetostn[1,:] += no_GPe

    edges = np.concatenate((STNtogpe,GPetostn),axis=1)
    
    
    gpe_inter = GPe_interconnections.astype(int)

    edges = np.concatenate((edges,gpe_inter),axis = 1)
    
    G = nx.DiGraph()
    G.add_nodes_from(range(no_STN+no_GPe)) 
    G.add_edges_from(np.transpose(edges))
    
    Reference = C_random_k_in_graph(no_STN+no_GPe, k, 10000*k*(no_STN+no_GPe))
    
    R = nx.DiGraph()
    R.add_nodes_from(range(no_STN+no_GPe)) 
    R.add_edges_from(np.transpose(Reference))
    
    RClust = average_clustering(R)
    RPath = average_shortest_path_length(R)
    
    GClust = average_clustering(G)
    GPath = average_shortest_path_length(G)
    
    
    SW = (GClust/RClust)/(GPath/RPath)
    
    print('Reference Clustering: ',RClust)
    print('Reference Path : ', RPath)
    print('Clustering: ',GClust)
    print('Path : ', GPath)
    print('Small world index: ', SW)

    GPe_pos = np.array( (2*np.ones(no_GPe),np.linspace(0,5,no_GPe)) )
    
    for i in range(no_GPe):
        GPe_pos[0,i] += 0.2*(i%5)
    
    STN_pos = np.array( (np.zeros(no_STN),np.linspace(1,4,no_STN)) )
    
    node_pos = {}
    for i in range(no_GPe):
        node_pos[i] = GPe_pos[:,i]
    for i in range(no_GPe,no_GPe+no_STN):
        node_pos[i] = STN_pos[:,i-no_GPe]

    #nx.write_gexf(G,'C:\\Educational stuff\\5th Year\\Thesis\\New Code Base\\'+name+'.gexf')
    if disp:
        AF.Show_Network_pos3(G,node_pos)


def GenerateNetwork(no_STN,no_GPe,GfS_in, SfG_in,GfG_in,alpha ):
    #change the config

    set_config(No_STN=no_STN,
    No_GPe= no_GPe,
    GfS_in= GfS_in,
    SfG_in= SfG_in,
    GfG_in= GfG_in,
    alpha=alpha)
    
    no_STN= int(no_STN)
    no_GPe= int(no_GPe)
    GfS_in= int(GfS_in)
    SfG_in= int(SfG_in)
    GfG_in= int(GfG_in)
    alpha= int(alpha)
    
    #the output is [from, to] but its defined by the in degree, i.e fixed [,to]
    
    t0 = runtime.time()
    STNtoGPe = K_In_Diff_Nets(no_STN,no_GPe, GfS_in, alpha)
    GPetoSTN = K_In_Diff_Nets(no_GPe,no_STN, SfG_in,alpha)
    GPe_interconnections = C_random_k_in_graph(no_GPe, GfG_in, alpha)
    t1 = runtime.time()
    
    print('Time taken to generate network: {:.2f} seconds'.format(t1-t0))
    
    # StG_dist = Connection_distribution(STNtoGPe,no_GPe,0)
    # GtS_dist = Connection_distribution(GPetoSTN,no_STN,0)
    # GtG_dist = Connection_distribution(GPe_interconnections,no_GPe,0)
    # 
    # print('STN to GPe distribution : \n',StG_dist[0:5,:])
    # print('GPe to STN distribution : \n',GtS_dist[0:5,:])
    # print('GPe to GPe distribution : \n',GtG_dist[0:5,:])
    
    #SW_R[i] = calc_SW(GPetoSTN,STNtoGPe,GPe,STN)
    
    #show_results(no_STN,no_GPe,GPe_interconnections,STNtoGPe,GPetoSTN,max(GfG_in,SfG_in,GfS_in),disp=True)
    
    np.save(path+'Intermediate Data\\fromSTNtoGpe.npy',STNtoGPe.astype(int))
    np.save(path+'Intermediate Data\\fromGPetoSTN.npy',GPetoSTN.astype(int))
    np.save(path+'Intermediate Data\\GPe Interconnections.npy',GPe_interconnections.astype(int))
    
    print('Network created, run the network by pressing run, or view the graph created in the folder')


def chkbtnstate(a):
    return 'selected' in a.state()

def set_config(No_STN,No_GPe,GfS_in,SfG_in,GfG_in,alpha):

    config = SafeConfigParser()
    config.read('config1.ini')  
    config.set('main', 'No_STN', str(No_STN))
    config.set('main', 'No_GPe', str(No_GPe))
    config.set('main', 'GfromG',str(GfG_in))
    config.set('main', 'GfromS', str(GfS_in))
    config.set('main', 'SfromG', str(SfG_in))
    config.set('main', 'alpha', str(alpha)) 
    with open('config1.ini', 'w') as f:
        config.write(f)
    
def set_default_config():
    
    config = SafeConfigParser()
    config.add_section('main')
    config.read('config1.ini')  
    config.set('main', 'No_STN', '100')
    config.set('main', 'No_GPe', '200')
    config.set('main', 'GfromG','5')
    config.set('main', 'GfromS', '5')
    config.set('main', 'SfromG', '5')
    config.set('main', 'alpha', '1') 
    with open('config1.ini', 'w') as f:
        config.write(f)
        
    

class STN_GPe_Wrapper():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("STN-GPe simulation")
        self.create_widgets()
        
        self.radio_variable = tk.StringVar()
        self.combobox_value = tk.StringVar()

    def create_widgets(self):
        # Create some room around all the internal frames
        self.window['padx'] = 5
        self.window['pady'] = 5
        
        frame_width = 35
  
        # - - - - - - - - - - - - - - - - - - - - -
        # The Commands frame
        # cmd_frame = ttk.LabelFrame(self.window, text="Commands", padx=5, pady=5, relief=tk.RIDGE)
        cmd_frame = ttk.LabelFrame(self.window, text="Connection Strengths", relief=tk.RIDGE)
        cmd_frame.grid(row=2, column=2, sticky=tk.E + tk.W + tk.N + tk.S, pady=5, padx=5)
        
        
        CTXtoSTNL = ttk.Label(cmd_frame, text="CTX to STN")
        CTXtoSTNL.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)
        CTXtoSTNL.config(width=frame_width)

        GPetoSTNL = ttk.Label(cmd_frame, text="GPe to STN")
        GPetoSTNL.grid(row=3, column=1,  sticky=tk.W, pady=5, padx=5)
        
        STNtoGPeL = ttk.Label(cmd_frame, text="STN to GPe")
        STNtoGPeL.grid(row=4, column=1, sticky=tk.W, pady=5, padx=5)
        
        GPetoGPeL = ttk.Label(cmd_frame, text="GPe to GPe")
        GPetoGPeL.grid(row=5, column=1,  sticky=tk.W, pady=5, padx=5)
        
        STRtoGPeL = ttk.Label(cmd_frame, text="STR to GPe")
        STRtoGPeL.grid(row=6, column=1,  sticky=tk.W, pady=5, padx=5)
        
        CTXtoSTN = ttk.Entry(cmd_frame, width=5)
        CTXtoSTN.grid(row=2, column=2, pady=5, padx = 20)
        CTXtoSTN.insert(tk.END, 2)

        GPetoSTN = ttk.Entry(cmd_frame, width=5)
        GPetoSTN.grid(row=3, column=2, pady=5, padx = 20)
        GPetoSTN.insert(tk.END, 0.1)
        
        STNtoGPe = ttk.Entry(cmd_frame, width=5)
        STNtoGPe.grid(row=4, column=2, pady=5, padx = 20)
        STNtoGPe.insert(tk.END, 0.5)
        
        GPetoGPe = ttk.Entry(cmd_frame, width=5)
        GPetoGPe.grid(row=5, column=2, pady=5, padx = 20)
        GPetoGPe.insert(tk.END, 0.5)
        
        STRtoGPe = ttk.Entry(cmd_frame, width=5)
        STRtoGPe.grid(row=6, column=2, pady=5, padx = 20)
        STRtoGPe.insert(tk.END, 1)

        #Inputs_Frame
        
        inp_frame = ttk.LabelFrame(self.window, text="Inputs", relief=tk.RIDGE)
        inp_frame.grid(row=2, column=1, sticky=tk.E + tk.W + tk.N + tk.S, pady=5, padx=5)
        r = 1
        STN_DC_offsetL = ttk.Label(inp_frame, text="STN DC offset (mV)")
        STN_DC_offsetL.grid(row=r, column=1, sticky=tk.W, pady=5, padx=5)
        STN_DC_offsetL.config(width=frame_width)
        
        STN_DC_offset = ttk.Entry(inp_frame, width=5)
        STN_DC_offset.grid(row=r, column=2, pady=5, padx = 20)
        STN_DC_offset.insert(tk.END, 0)
        r+=1
        
        GPe_DC_offsetL = ttk.Label(inp_frame, text="GPe DC offset (mV)")
        GPe_DC_offsetL.grid(row=r, column=1, sticky=tk.W, pady=5, padx=5)
        GPe_DC_offsetL.config(width=30)
        
        GPe_DC_offset = ttk.Entry(inp_frame, width=5)
        GPe_DC_offset.grid(row=r, column=2, pady=5, padx = 20)
        GPe_DC_offset.insert(tk.END, 0)
        r+=1
        
        CTX_FreqL = ttk.Label(inp_frame, text="Cortical Frequency (Hz)")
        CTX_FreqL.grid(row=r, column=1, sticky=tk.W, pady=5, padx=5)
        CTX_FreqL.config(width=30)
        
        CTX_Freq = ttk.Entry(inp_frame, width=5)
        CTX_Freq.grid(row=r, column=2, pady=5, padx = 20)
        CTX_Freq.insert(tk.END, 20)
        r+=1
        
        CTX_CorrL = ttk.Label(inp_frame, text="% of Synchronised Neurons")
        CTX_CorrL.grid(row=r, column=1, sticky=tk.W, pady=0, padx=20)
        CTX_CorrL.config(width=30)

        CTX_Corr = tk.Scale(inp_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                            width=8, length=100, resolution = 0.05)
        CTX_Corr.grid(row=r, column=2,  pady=0)
        r+=1
        
        CTX_PhaseL = ttk.Label(inp_frame, text="Phase")
        CTX_PhaseL.grid(row=r, column=1, sticky=tk.W, pady=0, padx=20)
        CTX_PhaseL.config(width=30)

        CTX_Phase = tk.Scale(inp_frame, from_=-3.1, to=3.1, orient=tk.HORIZONTAL,
                            width=8, length=100, resolution = 0.1)
        CTX_Phase.grid(row=r, column=2, padx = 0, sticky=tk.W)
        CTX_Phase.set(float(0))
        r+=1
        
        STR_FreqL = ttk.Label(inp_frame, text="Striatal Frequency (Hz)")
        STR_FreqL.grid(row=r, column=1, sticky=tk.W, pady=5, padx=5)
        STR_FreqL.config(width=30)
        
        STR_Freq = ttk.Entry(inp_frame, width=5)
        STR_Freq.grid(row=r, column=2, pady=5, padx = 20)
        STR_Freq.insert(tk.END, 3)
        r+=1
        
        STR_CorrL = ttk.Label(inp_frame, text="Synchronised")
        STR_CorrL.grid(row=r, column=1, sticky=tk.W, pady=0, padx=20)
        STR_CorrL.config(width=30)

        STR_Corr = tk.Scale(inp_frame, from_=0, to=1, orient=tk.HORIZONTAL,
                            width=8, length=100, resolution = 0.05)
        STR_Corr.grid(row=r, column=2,  pady=0)
        r+=1
        
        STR_PhaseL = ttk.Label(inp_frame, text="Phase")
        STR_PhaseL.grid(row=r, column=1, sticky=tk.W, pady=0, padx=20)
        STR_PhaseL.config(width=30)

        STR_Phase = tk.Scale(inp_frame, from_=-3.1, to=3.1, orient=tk.HORIZONTAL,
                            width=8, length=100, resolution = 0.1)
        STR_Phase.grid(row=r, column=2, padx = 0, sticky=tk.W)
        STR_Phase.set(float(0))
        r+=1
        
         ####Simulation frame
        ########################################################
        dur_frame = ttk.LabelFrame(self.window, text="Simulation", relief=tk.RIDGE)
        dur_frame.grid(row=3, column=2, sticky=tk.E + tk.W + tk.N + tk.S, pady=5, padx=5)
        
        i = 1
        
        timeL = ttk.Label(dur_frame, text="Time (ms)")
        timeL.grid(row=i, column=1, sticky=tk.W, pady=5, padx=5)
        timeL.config(width=frame_width)
        
        time = ttk.Entry(dur_frame, width=5)
        time.grid(row=i, column=2, pady=5, padx = 20)
        time.insert(tk.END, 1000)
        i+=1
        
        RasterL = ttk.Label(dur_frame, text="Raster Plots")
        RasterL.grid(row=i, column=1, sticky=tk.W, pady=5, padx=5)
        RasterL.config(width=30)

        Raster = ttk.Checkbutton(dur_frame)
        Raster.grid(row=i, column=2,  pady=0)
        i+=1
        
        LFPL = ttk.Label(dur_frame, text="LFP Plots")
        LFPL.grid(row=i, column=1, sticky=tk.W, pady=5, padx=5)
        LFPL.config(width=30)

        LFP = ttk.Checkbutton(dur_frame)
        LFP.grid(row=i, column=2,  pady=0)
        i+=1
        
        PowerSpectrumL = ttk.Label(dur_frame, text="Power Spectrum")
        PowerSpectrumL.grid(row=i, column=1, sticky=tk.W, pady=5, padx=5)
        PowerSpectrumL.config(width=30)

        PowerSpectrum = ttk.Checkbutton(dur_frame)
        PowerSpectrum.grid(row=i, column=2,  pady=0)
        i+=1
        
        CoherenceL = ttk.Label(dur_frame, text="Coherence")
        CoherenceL.grid(row=i, column=1, sticky=tk.W, pady=5, padx=5)
        CoherenceL.config(width=30)

        Coherence = ttk.Checkbutton(dur_frame)
        Coherence.grid(row=i, column=2,  pady=0)
        i+=1
        
        #plot_combo = (chkbtnstate(Raster),chkbtnstate(LFP),chkbtnstate(PowerSpectrum))
        
        Run = tk.Button(dur_frame, text="Run", width = 8, command = lambda: 
        STN_GPe(Cortex_gain=CTXtoSTN.get(),
        plots = (chkbtnstate(Raster),chkbtnstate(LFP),chkbtnstate(PowerSpectrum),chkbtnstate(Coherence)),
        GPe_gain=GPetoSTN.get(),
        STN_gain=STNtoGPe.get(),
        InterGPe_gain=GPetoGPe.get(),
        Striatal_gain=STRtoGPe.get(),
        time =time.get(),
        delay = (6/0.25),
        STN_DC_offset = STN_DC_offset.get(),
        GPe_DC_offset = GPe_DC_offset.get(),
        DBS = False, 
        cortex_frequency = CTX_Freq.get(), 
        cortex_correlated = CTX_Corr.get(),
        cortex_phase = CTX_Phase.get(),
        striatum_frequency = STR_Freq.get(), 
        striatum_correlated = STR_Corr.get(),
        striatum_phase = STR_Phase.get()))
        
        # myfunc(float(STN_GPe(CTXtoSTN.get(),GPetoSTN.get(),STNtoGPe.get(),GPetoGPe.get(),
        # STRtoGPe.get()))))

        bloatL = ttk.Label(dur_frame, text="")
        bloatL.grid(row=i, column=1, sticky=tk.W, pady=0, padx=5)
        bloatL.config(width=frame_width)
        i+=1
        
        Run.grid(row=i, column=1,sticky=tk.W, padx = 5, pady = 5)
        quit_button = tk.Button(dur_frame, text="Quit", command=self.window.destroy, width = 8)
        quit_button.grid(row=i, column=2, pady = 5, padx =5, sticky=tk.S)
        i+=1

        #Generation Frame
        ###################################################################################
        config = SafeConfigParser()
        config.read('config1.ini')
        
        gen_frame = ttk.LabelFrame(self.window, text="Network Generation", relief=tk.RIDGE)
        gen_frame.grid(row=3, column=1, sticky=tk.E + tk.W + tk.N + tk.S, pady=5, padx=5)
        
        r = 1
        No_STNL = ttk.Label(gen_frame, text="Number of STN Neurons")
        No_STNL.grid(row=r, column=1, sticky=tk.W, pady=5, padx=5)
        No_STNL.config(width=frame_width)
        
        No_STN = ttk.Entry(gen_frame, width=5)
        No_STN.grid(row=r, column=2, pady=3, padx = 20)
        No_STN.insert(tk.END, config.get('main', 'No_STN'))
        r+=1        
    
        No_GPeL = ttk.Label(gen_frame, text="Number of GPe Neurons")
        No_GPeL.grid(row=r, column=1, sticky=tk.W, pady=5, padx=5)
        No_GPeL.config(width=frame_width)
        
        No_GPe = ttk.Entry(gen_frame, width=5)
        No_GPe.grid(row=r, column=2, pady=3, padx = 20)
        No_GPe.insert(tk.END, config.get('main', 'No_GPe'))
        r+=1  
        
        GfromSL = ttk.Label(gen_frame, text="GPe in-degree from STN")
        GfromSL.grid(row=r, column=1, sticky=tk.W, pady=5, padx=5)
        GfromSL.config(width=frame_width)
        
        GfromS = ttk.Entry(gen_frame, width=5)
        GfromS.grid(row=r, column=2, pady=3, padx = 20)
        GfromS.insert(tk.END, config.get('main', 'GfromS'))
        r+=1  
        
        GfromGL = ttk.Label(gen_frame, text="GPe in-degree from GPe")
        GfromGL.grid(row=r, column=1, sticky=tk.W, pady=5, padx=5)
        GfromGL.config(width=frame_width)
        
        GfromG = ttk.Entry(gen_frame, width=5)
        GfromG.grid(row=r, column=2, pady=3, padx = 20)
        GfromG.insert(tk.END, config.get('main', 'GfromG'))
        r+=1  
        
        SfromGL = ttk.Label(gen_frame, text="STN in-degree from GPe")
        SfromGL.grid(row=r, column=1, sticky=tk.W, pady=5, padx=5)
        SfromGL.config(width=frame_width)
        
        SfromG = ttk.Entry(gen_frame, width=5)
        SfromG.grid(row=r, column=2, pady=3, padx = 20)
        SfromG.insert(tk.END, config.get('main', 'SfromG'))
        r+=1  
        
        alphaL = ttk.Label(gen_frame, text="Alpha")
        alphaL.grid(row=r, column=1, sticky=tk.W, pady=5, padx=5)
        alphaL.config(width=frame_width)
        
        alpha = ttk.Entry(gen_frame, width=5)
        alpha.grid(row=r, column=2, pady=3, padx = 20)
        alpha.insert(tk.END, config.get('main', 'alpha'))
        r+=1  

        Gen_Net = tk.Button(gen_frame, text="Generate Network", command = lambda: GenerateNetwork(no_STN = No_STN.get(),
        no_GPe = No_GPe.get(),
        GfS_in = GfromS.get(), 
        SfG_in = SfromG.get(),
        GfG_in = GfromG.get(),
        alpha = alpha.get()))
        
        Gen_Net.grid(row=14, column=1, sticky = tk.W, padx = 5, pady = 3)
        
        
        # phantom_label = tk.Label(self.window, text="")
        # phantom_label.grid(row=8, column=1, padx = 30)

        #this throws a soft error which interrupts the kernel but does not distrupt the program
        #a = tk.threrror()


#Create the entire GUI program
program = STN_GPe_Wrapper()

# Start the GUI event loop
program.window.mainloop()

# STN_GPe(Cortex_gain=0,
# GPe_gain=0.1,
# STN_gain=0.5,
# InterGPe_gain=0.5,
# Striatal_gain=0,
# time =1000,
# delay = (6/0.25),
# STN_DC_offset = 0,
# GPe_DC_offset = 0,
# DBS = False, 
# cortex_frequency = 20, 
# cortex_correlated = False, 
# cortex_phase = 0,
# striatum_frequency = 3, 
# striatum_phase = 0, 
# striatum_correlated = False, 
# plots=(1,1,0,0) )
