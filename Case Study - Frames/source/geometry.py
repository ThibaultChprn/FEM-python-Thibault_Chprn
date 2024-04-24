import numpy as np
import math 

class mesh:

    def __init__(self, name):
        self.name = name
        
    def getMesh(self):
        analysis = self.name

        if analysis=="Assignment_2023":
            self.getMesh_2023()
        elif analysis=="Assignment_2024":
            self.getMesh_2024()

    def getMesh_2024(self, VerticalDistance=0.5, HorizontalDistance=0.5):
        # Material and excitation properties - SI units convention (N,m,kg,s)
        # Elasticity Modulus
        E, rho = 210e9,7850

        # HEB 260 properties
        A_HEB, I_HEB = 11.8e-3, 149.2e-6 

        # ROR 193.7 x 25 properties 
        A_ROR, I_ROR = 13.2e-3, 0 #48e-6 

        #Create nodal coordinates from 0.0m to 4.0m every VerticalDistance (m)
        nodesV1 = np.arange(0,4.0,VerticalDistance)
        #Create nodal coordinates from 4.0m to 8.0m every VerticalDistance (m)
        nodesV2 = np.arange(4.0,8+VerticalDistance,VerticalDistance)
        #Stack these nodes together
        nodesV = np.hstack([nodesV1,nodesV2])
        #Create nodal coordinates from 0.0m to 5.0m every HorizontalDistance (m)
        nodesH = np.arange(0,5.1,HorizontalDistance)
        
        #Create the matrix that stores the nodal coordinates
        nodes = np.zeros((np.shape(nodesV)[0]+np.shape(nodesH)[0]-1,2))
        nodes[0:np.shape(nodesV)[0],1]=nodesV
        nodes[np.shape(nodesV)[0]:,0]=nodesH[1:]
        nodes[np.shape(nodesV)[0]:,1]=8.

        self.nodes = nodes
        
        numberOfDofs = nodes.shape[0]*3 
        numberOfElements = nodes.shape[0]

        elements = np.arange(0,numberOfElements,1)
        connectivity=np.zeros((numberOfElements,2))
        connectivity[:,0]=elements
        connectivity[:,1]=elements+1
        connectivity[-1,:]=[nodesV1.shape[0],nodes.shape[0]-1]

        self.connectivity = connectivity


        self.ElementProperties = np.array([
            [E, rho, A_HEB, I_HEB, 0],
            [E, rho, A_ROR, I_ROR, 1]
        ])

        crossSections = np.zeros_like(connectivity)
        crossSections = crossSections[:,0]
        crossSections[-1]=1
        self.crossSections = np.reshape(crossSections.astype(int),(27,1))

        self.BCs = np.array([
            [0,1,1,1]
        ])   

        self.nodalLoads=np.array([
                [1, 0, 0, 0],
                [2, 0, 0, 0]
            ])

        self.distributedLoads = np.zeros((numberOfElements,2))

        self.BCsRot = np.array([
                [0, 0]
            ])

    def getMesh_2023(self):
        # Material and excitation properties - SI units convention (N,m,kg,s)
        # Elasticity Modulus
        E, rho = 210e9, 7850
        # HEB 260 properties
        A_HEB, I_HEB = 11.8e-3, 149.2e-6 
        # ROR 193.7 x 25 properties 
        A_ROR, I_ROR = 13.2e-3, 48e-6 
        # Rotational spring stiffness
        kr, ks = 50e6, 0.6e9
        P1, P2, q1, q2= 75e3, 150e3, 2.5e3, 10e3 

        self.nodes = np.array([
            [0, 0],
            [4, 10],
            [4, 5],
            [4, 0],
            [8, 5],
            [12, 5],
            [16, 5],
            [19,5],
            [4, 0],# Virtual duplicate node, same coordinates as Node 4
            [19,5] # Virtual duplicate node, same coordinates as Node 8
        ])

        self.connectivity = np.array([
                [0, 1],
                [1, 2],
                [2, 3],
                [1,4],
                [1,5],
                [1,6],
                [2, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [3, 8], #Virtual truss element representing the rot. spring
                [7, 9] #Virtual truss element representing the spring
            ])

        self.nodalLoads = np.array([
                [1, P1, 0, 0],
                [2, P2, 0, 0]
            ])

        self.distributedLoads = np.array([
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [0,0],
                [-q2,-q2], #[-q2, -(q1+(q2-q1)/4*3)],
                [-q2,-q2], #[-(q1+(q2-q1)/4*3),-(q1+(q2-q1)/4*2)],
                [-q2,-q2], #[-(q1+(q2-q1)/4*2),-(q1+(q2-q1)/4*1)],
                [-q2,-q2], #[-(q1+(q2-q1)/4*1),-q1],     
                [0,0],
                [0,0]
            ])

        self.BCs = np.array([
                [0,1,1,1], # Inclined support condition (expressed in "local" coordinates) - Rotation is constrained to avoid singularities
                [3,1,1,0],
                [7,0,1,1], # Roller support condition - Rotation is constrained to avoid singularities (zero in the diagonal)
                [8,1,1,1], # Virtual node is fully constrained as it represents the ground
                [9,1,1,1], # Virtual node is fully constrained as it represents the ground    
            ])

        self.BCsRot = np.array([
                [0, math.atan(10/4)]
            ])

        self.ElementProperties = np.array([
                [E, rho, A_HEB, I_HEB, 0], # Inclined support condition (expressed in "local" coordinates) - Rotation is constrained to avoid singularities
                [E, rho, A_ROR, I_ROR, 1],
                [0, 0, 0, kr, 2], # Roller support condition - Rotation is constrained to avoid singularities (zero in the diagonal)
                [0, ks, 0, 0, 3], # Virtual node is fully constrained as it represents the ground
                [E, rho, A_HEB, I_HEB, 4], # Virtual node is fully constrained as it represents the ground    
            ])

        self.crossSections = np.array([
                [1], #Index=1 to specify truss elements
                [0], #Index=0 to specify beam elements
                [0],
                [1],
                [1],
                [1],
                [0],
                [0],
                [0], 
                [4], #Index=4 to specify the beam element with a moment release
                [2], #Index=2 to specify the rotational spring element
                [3]  #Index=3 to specify the translational spring element
            ])
            

            

   