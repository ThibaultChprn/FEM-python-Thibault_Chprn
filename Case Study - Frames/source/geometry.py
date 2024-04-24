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
        elif analysis=="Assignment_2022":
            self.getMesh_2022()
        elif analysis=="PlaneFrameDemo":
            self.getMesh_Frame()

    def getMesh_2022(self):
        E, rho = 2.1e11, 7850
        A1, I1 = 400e-3,  1450e-6
        A2, I2 =  250e-3,  980e-6
        P1,P2,P3 = [250e3, 80e3, 54e3]
        ks = 5e3

        self.nodes = np.array([
            [0, 0],
            [9, 0],
            [10.5, 0],
            [12, 0],
            [21, 0],
            [3, -5],
            [18, -5],
            [21, 0] # Virtual duplicate node, same coordinates as Node 4
        ])

        self.connectivity = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [3, 6],
            [4, 7] #Virtual element representing the spring "truss"
        ])

        self.ElementProperties = np.array([
            [E, rho, A1, I1, 0],
            [E, rho, A2, I2, 0],
            [0,  0, ks, 0 , 3],
        ])

        self.crossSections = np.array([
            [0],
            [0],
            [0],
            [0],
            [1],
            [1],
            [2]
        ])

        phi = np.pi/4
        cos45 = np.cos(phi)
        sin45 = np.sin(phi)

        self.nodalLoads = np.array([
            [1, P2*cos45, -P2*sin45, 0],
            [2, 0, -P1, 0],
            [3, -P3*cos45, -P3*sin45, 0]
        ])

        self.BCs = np.array([
            [0,1,1,0],
            [5,1,0,0], # Inclined support condition (left - expressed in "local" coordinates - implies that after rotating localy the x-axis is restrained )
            [6,1,1,1], # Inclined support condition (right)
            [7,1,1,1], # Virtual node needs to be fully constrained as it represents the ground    
        ])

        self.BCsRot = np.array([
                [5, -math.atan(5/6) ],
                [6, 0 ]
        ])

        self.distributedLoads = np.zeros((self.connectivity.shape[0],2))


    def getMesh_Frame(self):
        E = 4e10    #Young modulus 
        rho = 4500  #Density

        A1, I1 = 102e-3,  980e-6 #Cross-section area of elements with type 1
        A2, I2 =  93e-3,  720e-6
        A3, I3 = 140e-3, 2430e-6

        self.nodes = np.array([
            [0, 0], #1st row contains the x,y coordinates of the 1st node. Pay attention to the [x,y] structure of the array
            [6, 0],
            [0, 6],
            [6, 6],
            [0, 12],
            [6, 12]
        ])

        self.connectivity = np.array([
            [0, 2], #1st element in 1st row. The start node is N1, so the first node, so index 0. The end node is N3, so index 2.
            [1, 3],
            [2, 4],
            [3, 5],
            [2, 3],
            [4, 5]    
        ])

        self.ElementProperties = np.array([
            [E, rho, A1, I1, 0],
            [E, rho, A2, I2, 0],
            [E, rho, A3, I3, 0]
        ])

        self.crossSections = np.array([
            [0], #1st row corresponds to the first element. We want its cross-section, so A1, I1. A1.I1 are in the 1st row of the crossSectionProperties matrix. So the index is 0.
            [0], #So, for each element we will have actual cross section = crossSectionProperties[crossSections[ElementIndex]][0]
            [1], 
            [1],
            [2], 
            [2]    
        ])

        self.BCs = np.array([
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ])

        self.nodalLoads=np.zeros((self.nodes.shape[0],4))

        self.distributedLoads = np.zeros((self.connectivity.shape[0],2))

        self.BCsRot = np.array([
                [0, 0]
            ])
        


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
            

            

   