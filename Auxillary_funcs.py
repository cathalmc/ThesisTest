import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



def Raster_plot(act,step):
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
            
    plt.imshow(act2,origin='lower',interpolation = 'none',aspect = 'auto', extent = [0,time*0.25,0,length])
    plt.title('Raster plot for ' + str(length) + ' neurons')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neurons')
    plt.show()



# def Raster_plot_Simple(act):
#     time = np.size(act[0,:])
#     length = np.size(act[:,0])
#     step = 1
#     
#     act2 = np.zeros((sc*length,int(time/step)))
# 
#     
#     for i in range(length):
#         for j in range(int(sc/2)):
#             act2[sc*i+j,:] = act[i,0:time:step]
#             
#     plt.imshow(act2,origin='lower')
#             
def Show_Network_and_Outputs(t,Nets,dat,positons=0): 
    
    """ Plot network diagrams and activations
        t is the time vector
        Nets is a list of the networks to be displayed,
        dat is the corresponding data for that network
        positions are the node locations
        """

    
    max_neur = 0
    for i in range(len(Nets)):
        if Nets[i].number_of_nodes() >max_neur:
            max_neur = Nets[i].number_of_nodes()
    
    #Generate the colours for each node and activation
    #Its probably easier in general to calculate the colours for the largest
    #network and then for smaller nets just use the colours needed, as opposed
    #to creating a colour vector for each neuron
    col = []
    for i in range(max_neur):
        col.append('C'+str(i%10))
    
    
    #main loop for plotting
    #plots network in left column, data in right
    plt.figure(1)
    for i in range(len(Nets)):
        #edgewidth = [ d['weight'] for (u,v,d) in Nets[1].edges(data=True)]
        #Generate the numbers for each neuron
        labels = {}
        for j in range(Nets[i].number_of_nodes()):
            labels[j] = str(j)
        
        #positions for each node, kamada is the shortest path algorithm, it looks the best
        if type(positons) is not list or not tuple:
            pos = nx.kamada_kawai_layout(Nets[0])
        else:
            pos = positons[i]
        
        #plot network in left column
        plt.subplot( len(Nets)*100 + 20 + 2*i + 1)
        
            
        nx.draw(Nets[i],pos,node_size = 40, node_color=col)#,width=edgewidth)
        nx.draw_networkx_labels(Nets[i], pos, labels, font_size=6)
        
            

        #plot data in right column
        plt.subplot( len(Nets)*100 + 20 + 2*i + 2)
        #plt.ylim((-100, 40))
        for j in range(np.size(dat[i],0)):
            plt.plot(t,dat[i][j,:],col[j])
    
    plt.show()
    
def Show_Network(Net):
    
    col = []
    for i in range(Net.number_of_nodes()):
        col.append('C'+str(i%10))
    plt.figure(1)
    pos = nx.kamada_kawai_layout(Net)
    #pos = nx.shell_layout(Net)
    
    labels = {}
    for i in range(Net.number_of_nodes()):
        labels[i] = str(i)
    nx.draw(Net,pos,node_size = 40, node_color=col)
    nx.draw_networkx_labels(Net, pos, labels, font_size=6)
    plt.show()
    
def Show_Network_pos2(Net,pos):
    
    edgewidth = [ d['weight'] for (u,v,d) in Net.edges(data=True)]
    
    col = []
    for i in range(Net.number_of_nodes()):
        col.append('C'+str(i%10))
    plt.figure(1)
    
    labels = {}
    for i in range(Net.number_of_nodes()):
        labels[i] = str(i)
    
    #G contains weight values in the edges
    nx.draw(Net,pos,node_size = 40, node_color=col,width=edgewidth)
    nx.draw_networkx_labels(Net, pos, labels, font_size=6)
    plt.show()
    
def Show_Network_pos(Net,pos):
    node_pos = {}
    sh = np.shape(pos)
    for i in range(sh[0]):
            node_pos[i] = pos[i,:]
    
    
    col = []
    for i in range(Net.number_of_nodes()):
        col.append('C'+str(i%10))
    plt.figure(1)
    
    labels = {}
    for i in range(Net.number_of_nodes()):
        labels[i] = str(i)
    
    #G contains weight values in the edges
    nx.draw(Net,node_pos,node_size = 40, node_color=col)
    nx.draw_networkx_labels(Net, pos, labels, font_size=0)
    plt.show()
    
def Show_Network_posGPe(Net,pos,no_reg):
    
    node_pos = {}
    sh = np.shape(pos)
    
    num_per_reg = int(sh[0]/no_reg)
    
    for i in range(sh[0]):
            node_pos[i] = pos[i,:]
    
    
    col = []
    labels = {}
    for i in range(no_reg):
        for j in range(num_per_reg):
            col.append('C'+str(i%10))
            labels[i*num_per_reg+j] = i
            
    plt.figure(1)
    
    edgewidth = 0.3*np.ones(Net.number_of_edges())
    
    
    #G contains weight values in the edges
    nx.draw(Net,node_pos,node_size = 20, node_color=col,width = edgewidth)
    nx.draw_networkx_labels(Net, pos, labels, font_size=4)
    plt.show()


    
def Show_Outputs_old(t,y):
    col = []
    for i in range(np.size(y,0)):
        col.append('C'+str(i%10))
    
    for i in range(np.size(y,0)):
        plt.plot(t,y[i,:],col[i])
 
    plt.show()
    
    

def Show_Outputs(t,dat):
    
    plt.figure(1)
    for i in range(len(dat)):
        
        plt.subplot( len(dat)*100 + 10 + i + 1)
    
        for j in range(np.size(dat[i],0)):
            plt.plot(t,dat[i][j,:])
    
    plt.show()
