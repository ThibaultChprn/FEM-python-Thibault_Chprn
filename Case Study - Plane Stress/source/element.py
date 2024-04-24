import numpy as np


class Element:

    def __init__(self, material, section):
        self.material = material
        self.section = section

    #Function to compute the dof numbers of an element
    def element_dofs(self,nodes):
        #Initialization
        dofs=np.zeros((2*len(nodes),1))
        
        #x-displacement dofs
        dofs[0::2,0]=2*(nodes-1)
        #y-displacement dofs
        dofs[1::2,0]=2*nodes-1
        
        dofs = dofs.astype(int)
        return dofs
    
    def Bf(self,Nrev, Nsev, nodes):
        #Nr, Ns are the shape function derivatives with the respect
        #to the isoparametric coordinates evaluated at some point
        
        #Jacobian matrix, Jacobian determinant and Jacobian matrix inverse
        J = (np.array([(Nrev), (Nsev)])[:,:,0]).dot(nodes)
        detJ = np.linalg.det(J)
        G = np.linalg.inv(J)
        
        #Shape function derivatives with respect to x and y
        Nx = G[0,:].dot((np.array([(Nrev), (Nsev)])[:,:,0]))
        Ny = G[1,:].dot((np.array([(Nrev), (Nsev)])[:,:,0]))

        #Initialize B function
        B = np.zeros((3,2*len(Nrev)))
        
        #Loop over nodes and assemble B function
        for k in range(len(Nrev)):
            B[:,2*k:(2*k+2)]=np.array([
                [Nx[k], 0],
                [0,     Ny[k]],
                [Ny[k], Nx[k]]
            ])
        
        Betas={}
        Betas['Beta']=B
        Betas['detJ']=detJ
        return Betas
    
    def surf_element_matrix(self, nodes, C, Nr, Ns, points, weights):
        #Initialization
        nodes_number = np.shape(nodes)[0]
        K = np.zeros((2*nodes_number,2*nodes_number))
        
        if type(weights)==float:
            loop=1        
        else:
            loop=len(weights)
                
        for k in range(loop):
            #Gauss point coordinates
            if type(weights)==float:
                r = points[0]
                s = points[1]
            else:
                r = points[k][0]
                s = points[k][1]
            
            
            #Evaluate shape function derivatives
            Nri = Nr(r,s)
            Nsi = Ns(r,s)
            
            #B function and Jacobian determinant
            Betas = self.Bf( Nri, Nsi, nodes )
            B = Betas['Beta']
            detJ = Betas['detJ']
            
            #Add Gauss point contribution to element stiffness matrix
            if type(weights)==float:
                K = K + weights*B.T.dot(C).dot(B).dot(detJ)
            else:
                K = K + weights[k]*B.T.dot(C).dot(B).dot(detJ)
            
        
        return K
    
    #Function to compute the load vector of plane solid element edges
    def edge_element_load(self, nodes, P, h, I):
        
        nodes_number = np.shape(nodes)[0]
        
        if nodes_number==2:
            #2 noded linear edge
            N = lambda r : np.array([
                [0.5*(1-r)],
                [0.5*(1+r)]])
            
            Nr = lambda r: np.array([
                [-0.5],
                [0.5]])
            
            weights = np.array([
                [1],
                [1]])
            
            points = np.array([
                [-1/np.sqrt(3)],
                [1/np.sqrt(3)] 
            ]) 
            
        elif nodes_number==3:
            #3 noded quadratic edge
            
            N = lambda r : np.array([
                [0.5*r*(r-1)],
                [1-r**2],
                [0.5*r*(r+1)]])
            
            Nr = lambda r: np.array([
                [r-0.5],
                [-2*r],
                [r+0.5]])
            
            points = np.array([
                [-np.sqrt(3/5)],
                [0],
                [np.sqrt(3/5)]])
            
            weights = np.array([
                [5/9],
                [8/9],
                [5/9]
            ]) 
            
        #Initialization
        f = np.zeros((2*nodes_number,1))
        
        for k in range(len(weights)):
            #Gauss point coordinates
            r = points[k]
            
            #Evaluate shape function at Gauss point
            Ni = N(r)
            Ni = np.squeeze(Ni, axis=1)
            Ni = np.squeeze(Ni, axis=1)
            
            #Convert shape function to vector
            Niv = np.zeros((2,2*len(Ni)))
            Niv[0,0::2] = Ni
            Niv[1,1::2] = Ni
            
            #Evaluate shape function derivative at Gauss point
            Nri = Nr(r)
            
            #Jacobian determinant for the edge
            detJ = np.linalg.norm(Nri.T.dot(nodes))
            
            #Gauss point coordinates in the global system
            Ci = Ni.T.dot(nodes)
            
            #Evaluate traction at the Gauss point
            Fi = Element.traction(Ci[0] , Ci[1], P,h,I)
            
            #Add Gauss point contribution to edge load vector
            f = f + weights[k]*Niv.T.dot(Fi).dot(detJ)
            
        return f
    
    def traction(x,y,P,h,I):
        t = np.array([
            [0],
            [-P/(2*I)*(0.25*h**2-y**2)]
        ]
        )
        
        return t

    

class Quad4element(Element):

    _nodes = 4

    def __init__(self, material, section):
        super().__init__(material, section)


    def getShapeFunctions(self):

        #4 noded quadrilateral element
        N = lambda r,s: 0.25*np.array([[(1-r)*(1-s)],[(1+r)*(1-s)],[(1+r)(1-s)],[(1-r)*(1+s)]])
        Nr = lambda r,s: 0.25*np.array([[s-1],[1-s],[1+s],[-1-s]])
        Ns = lambda r,s: 0.25*np.array([[-1+r],[-1-r],[1+r],[1-r]])
        
        weights = np.array([[1],[1],[1],[1]])
        points = np.array([
            [-1/np.sqrt(3), -1/np.sqrt(3)],
            [1/np.sqrt(3), -1/np.sqrt(3)],
            [1/np.sqrt(3),  1/np.sqrt(3)],
            [-1/np.sqrt(3),  1/np.sqrt(3)]])

        #weights = 0.5
        #points = np.array([[1/3],[1/3]])
        
        return N, Nr, Ns, points, weights
    

class Tri3element(Element):

    _nodes = 3

    def __init__(self, material, section):
        super().__init__(material, section)

    def getShapeFunctions(self):

        #3 noded triangular element
        N = lambda r,s: np.array([[r],[s],[1-r-s]])            
        Nr = lambda r,s: np.array([[1],[0],[-1]])
        Ns = lambda r,s: np.array([[0],[1],[-1]])
        
        weights = 0.5
        points = np.array([[1/3],[1/3]])

        return N, Nr, Ns, points, weights
    
class Tri6element(Element):

    _nodes = 6

    def __init__(self, material, section):
        super().__init__(material, section)

    def getShapeFunctions(self):

        #6 noded triangular element
        N = lambda r,s: np.array([
            [(r+s-1)*(2*r+2*s-1)],
            [r*(2*r-1)],
            [s*(2*s-1)], 
            [4*r*(r+s-1)],
            [4*r*s],
            [4*s*(r+s-1)]
        ])
        
        Nr = lambda r,s: np.array([
            [-3+4*r+4*s],
            [4*r-1],
            [0], 
            [-4*(2*r+s-1)],
            [4*s],
            [-4*s]
        ])
        
        Ns = lambda r,s: np.array([
            [-3+4*r+4*s],
            [0],
            [4*s-1], 
            [-4*r],
            [4*r],
            [-4*(r+2*s-1)]
        ])
        
        weights = np.array([[1/6], [1/6], [1/6]])
        points = np.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3]])

        return N, Nr, Ns, points, weights
        
class Quad9element(Element):

    _nodes = 9

    def __init__(self, material, section):
        super().__init__(material, section)

    def getShapeFunctions(self):

        #9 noded quadrilateral element
        N = lambda r,s: np.array([
            [0.25*(r**2-r)*(s**2-s)],
            [0.25*(r**2+r)*(s**2-s)],
            [0.25*(r**2+r)*(s**2+s)],
            [0.25*(r**2-r)*(s**2+s)],
            [0.5 *(s**2-s)*(1-r**2)],
            [0.5 *(r**2+r)*(1-s**2)],
            [0.5 *(s**2+s)*(1-r**2)],
            [0.5 *(r**2-r)*(1-s**2)],
            [(1-r**2)*(1-s**2)]])
               
        Nr = lambda r,s: np.array([
            [0.25*(2*r-1)*(s**2-s)],
            [0.25*(2*r+1)*(s**2-s)],
            [0.25*(2*r+1)*(s**2+s)],
            [0.25*(2*r-1)*(s**2+s)],
            [0.5 *(s**2-s)*(-2*r)],
            [0.5 *(2*r+1)*(1-s**2)],
            [0.5 *(s**2+s)*(-2*r)],
            [0.5 *(2*r-1)*(1-s**2)],
            [(-2*r)*(1-s**2)]])
                
        Ns = lambda r,s: np.array([
            [0.25*(r**2-r)*(2*s-1)],
            [0.25*(r**2+r)*(2*s-1)],
            [0.25*(r**2+r)*(2*s+1)],
            [0.25*(r**2-r)*(2*s+1)],
            [0.5 *(2*s-1)*(1-r**2)],
            [0.5 *(r**2+r)*(-2*s)],
            [0.5 *(2*s+1)*(1-r**2)],
            [0.5 *(r**2-r)*(-2*s)],
            [(1-r**2)*(-2*s)]])
                
        weights = np.array([
            [25/81],
            [40/81],
            [25/81],
            [40/81],
            [64/81],
            [40/81],
            [25/81],
            [40/81],
            [25/81]]        
        )
        
        points = np.array([
            [-np.sqrt(3/5), -np.sqrt(3/5)],
            [0,     -np.sqrt(3/5)],
            [np.sqrt(3/5), -np.sqrt(3/5)],
            [-np.sqrt(3/5),     0],
            [0, 0],
            [np.sqrt(3/5),     0],
            [-np.sqrt(3/5), np.sqrt(3/5)],
            [0,       np.sqrt(3/5)],
            [np.sqrt(3/5),  np.sqrt(3/5)]            
        ])

        return N, Nr, Ns, points, weights
        