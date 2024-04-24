import numpy as np
import scipy.io


def getMesh(analysis="Quads4"):

    if analysis == "Quads4":
        mesh = scipy.io.loadmat('./source/Inputs_Quads1.mat')
        mesh['element_type']="Quad4"
    elif analysis == "Quads9":
        mesh = scipy.io.loadmat('./source/Inputs_Quads2.mat')
        mesh['element_type']="Quad9"
    elif analysis == "Tris6":
        mesh = scipy.io.loadmat('./source/Inputs_Tri2_Coarse.mat')
        mesh['element_type']="Tri6"
    elif analysis == "Tris":
        mesh = scipy.io.loadmat('./source/Inputs_Tri_Coarse.mat')
        mesh['element_type']="Tri3"
    else:
        mesh = scipy.io.loadmat('./source/Inputs_Quads1.mat')
        mesh['element_type']="Quad4"

    #Node coordinates -> mesh['nodes']
    #Element connectivity -> mesh['elements']
    #Nodes of the left bounded edge -> mesh['nodes_l']
    #Nodes of the right bounded edge -> mesh['edges_r']

    # Restrained degree-of-freedom labels
    restrainedDofs = np.array([
        2*mesh['nodes_l'].T-2,
        2*mesh['nodes_l'].T-1, 
    ]
    )

    restrainedDofs = np.squeeze(restrainedDofs, axis=1)
    restrainedDofs=restrainedDofs.reshape((6, ))
    restrainedDofs=restrainedDofs.astype(int)
    restrainedDofs = np.sort(restrainedDofs)

    BCs = {}
    BCs['essential'] = np.zeros((np.shape(restrainedDofs)[0],2))
    BCs['essential'][:,0]=restrainedDofs
    
    return mesh, BCs

    

#Function to compute the dof numbers of an element
def element_dofs(nodes):
    #Initialization
    dofs=np.zeros((2*len(nodes),1))
    
    #x-displacement dofs
    dofs[0::2,0]=2*(nodes-1)
    #y-displacement dofs
    dofs[1::2,0]=2*nodes-1
    
    dofs = dofs.astype(int)
    return dofs

