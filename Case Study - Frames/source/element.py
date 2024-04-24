import numpy as np


class Element:

    def __init__(self, material, section):
        self.material = material
        self.section = section

    def LocalBeamCondense(k_local, f_local, retained_dofs=[0,1,2,3,4]):
    
        #Initialize matrix
        ke_local = np.zeros((6,6))
        fe_local = np.zeros((6,1))

        #Apply static condensation
        alldofs = [0,1,2,3,4,5]
        condensed_dofs = np.setdiff1d(alldofs, retained_dofs)

        Krr = k_local[retained_dofs,:][:,retained_dofs]
        Krc = k_local[retained_dofs,:][:,condensed_dofs]
        Kcr = k_local[condensed_dofs,:][:,retained_dofs]
        Kcc = k_local[condensed_dofs,:][:,condensed_dofs]
        Fr = f_local[retained_dofs]
        Fc = f_local[condensed_dofs]
        
        ke_local_temp = Krr - Krc.dot(np.linalg.inv(Kcc)).dot(Kcr)
        fe_local_temp = Fr - Krc.dot(np.linalg.inv(Kcc)).dot(Fc)
        
        ke_local[retained_dofs,0:5]=ke_local_temp
        fe_local[retained_dofs,:] = fe_local_temp
        
        return ke_local, fe_local


class BeamElement_Gauss(Element):

    def __init__(self, material, section):
        super().__init__(material, section)

    def LocalK(self, L):       
        E,A,I = self.material.E, self.section.A, self.section.I
        #Define number of Gauss integration points needed
        npoints = 2
        points = np.polynomial.legendre.leggauss(npoints) #--> Number inside function defines the number of Gauss points
        #Evaluate Gauss point coordinates and weights
        GaussCoords,GaussWeights = points[0],points[1]
        ke_local=np.zeros((6,6))
        k_bending=np.zeros((4,4))
        
        #Loop over Gauss points (stiffness matrix assembly)
        for weight, xi in zip(GaussWeights,GaussCoords):
                
            #Form the shape function matrix 
            N2, N3, N5, N6 = 1/4*((1-xi)**2)*(2+xi), L/8*((1-xi)**2)*(1+xi),1/4*((1+xi)**2)*(2-xi), -L/8*((1+xi)**2)*(1-xi)
            dN2, dN3, dN5, dN6 = 6*xi/L**2, (3*xi-1)/L, -6*xi/L**2, (3*xi+1)/L

            N = np.array([ [N2], [N3], [N5], [N6] ])
            Bb = np.array([ [dN2], [dN3], [dN5], [dN6]])
            
            #Determine the Jacobian of the isoparametric mapping
            detJ = L/2
            
            #Evaluate strain-displacement matrix at Gauss point
            k_bending = k_bending + Bb.dot(E*I).dot(Bb.T)*weight*detJ
                    
            ke_local[0,0]= E*A/L
            ke_local[0,3]= -E*A/L
            ke_local[3,0]= -E*A/L
            ke_local[3,3]= E*A/L
            dofs=[1,2,4,5]
            ke_local[dofs,1:3]=k_bending[:,0:2]
            ke_local[dofs,4:6]=k_bending[:,2:4]
            
        return ke_local
    
    def LocalM(self,L):
        if self.section.I>0:
            r,A,I = self.material.rho, self.section.A, self.section.I
            m = r*A*L/420*np.array([
                    [ 140,     0,       0,  70,     0,       0],
                    [   0,   156,    22*L,   0,    54,   -13*L],
                    [   0,  22*L,  4*L**2,   0,  13*L, -3*L**2],
                    [  70,     0,       0, 140,     0,       0],
                    [   0,    54,    13*L,   0,   156,   -22*L],
                    [   0, -13*L, -3*L**2,   0, -22*L,  4*L**2],
            ])
        else:
            r,A = self.material.rho, self.section.A
            m = r*A*L/420*np.array([
                    [ 140,     0,       0,  70,     0,       0],
                    [   0,   0,    0,   0,    0,   0],
                    [   0,  0,  0,   0,  0, 0],
                    [  70,     0,       0, 140,     0,       0],
                    [   0,    0,    0,   0,   0,   0],
                    [   0, 0, 0,   0, 0,  0],
            ])

        return m
    
    def LocalF(self, q_start,q_end,L):
        q=(q_start+q_end)/2
        f = np.array([ 
                [0],
                [q*L/2],
                [q*L**2/12],
                [0],
                [q*L/2],
                [-q*L**2/12],
        ])
        return f   



class BeamElement_Spring(Element):

    def __init__(self, material, section):
        super().__init__(material, section)


    def LocalK(self, L):
        ks_x, ks_y, kr = self.material.rho, self.section.A, self.section.I
        
        k = np.array([
            [  ks_x, 0, 0, -ks_x, 0, 0 ],
            [  0, ks_y, 0, 0, -ks_y, 0 ],
            [  0, 0, kr, 0, 0, -kr ],
            [  -ks_x, 0, 0, ks_x, 0, 0 ],
            [  0, -ks_y, 0, 0, ks_y, 0 ],
            [  0, 0, -kr, 0, 0, kr ]
        ])
        return k
    
    def LocalM(self, L):
        m = np.ones((6,6))
        return m
    
    def LocalF(self, q_start=0,q_end=0,L=0):
        f = np.zeros((6,1))
        return f

class BeamElement_Condensed(Element):

    def __init__(self, material, section):
        super().__init__(material, section)

    def LocalK_DSM(self, L):
        E,A,I = self.material.E, self.section.A, self.section.I
        k = np.array([
                [  E*A/L,            0,           0, -E*A/L,            0,           0],
                [      0,  12*E*I/L**3,  6*E*I/L**2,      0, -12*E*I/L**3, 6*E*I/L**2],
                [      0,   6*E*I/L**2,     4*E*I/L,      0,  -6*E*I/L**2,     2*E*I/L],
                [ -E*A/L,            0,           0,  E*A/L,            0,           0],
                [      0, -12*E*I/L**3, -6*E*I/L**2,      0,  12*E*I/L**3, -6*E*I/L**2],
                [      0,   6*E*I/L**2,     2*E*I/L,      0,  -6*E*I/L**2,     4*E*I/L],
        ])


        return k
    
    #Here we assume a constant load over each element modeled as q=(q_start+q_end)/2
    def LocalF_DSM(self, q_start,q_end,L):
        q=(q_start+q_end)/2
        f = np.array([ 
                [0],
                [q*L/2],
                [q*L**2/12],
                [0],
                [q*L/2],
                [-q*L**2/12],
        ])
        return f
    
    def LocalM(self,L):
        if self.section.I>0:
            r,A,I = self.material.rho, self.section.A, self.section.I
            m = r*A*L/420*np.array([
                    [ 140,     0,       0,  70,     0,       0],
                    [   0,   156,    22*L,   0,    54,   -13*L],
                    [   0,  22*L,  4*L**2,   0,  13*L, -3*L**2],
                    [  70,     0,       0, 140,     0,       0],
                    [   0,    54,    13*L,   0,   156,   -22*L],
                    [   0, -13*L, -3*L**2,   0, -22*L,  4*L**2],
            ])
        else:
            r,A = self.material.rho, self.section.A
            m = r*A*L/420*np.array([
                    [ 140,     0,       0,  70,     0,       0],
                    [   0,   0,    0,   0,    0,   0],
                    [   0,  0,  0,   0,  0, 0],
                    [  70,     0,       0, 140,     0,       0],
                    [   0,    0,    0,   0,   0,   0],
                    [   0, 0, 0,   0, 0,  0],
            ])

        return m
    
    def LocalK(self, L):
        #Initialize matrix
        ke_local = np.zeros((6,6))
        #Evaluate local element stiffness matrix without any condensation
        k_local = self.LocalK_DSM(L)
        #Apply static condensation
        alldofs = [0,1,2,3,4,5]
        retained_dofs = [0,1,2,3,4]
        condensed_dofs = np.setdiff1d(alldofs, retained_dofs)

        Krr = k_local[retained_dofs,:][:,retained_dofs]
        Krc = k_local[retained_dofs,:][:,condensed_dofs]
        Kcr = k_local[condensed_dofs,:][:,retained_dofs]
        Kcc = k_local[condensed_dofs,:][:,condensed_dofs]
        
        ke_local_temp = Krr - Krc.dot(np.linalg.inv(Kcc)).dot(Kcr)
        
        ke_local[retained_dofs,0:5]=ke_local_temp
    
        return ke_local
    
    def LocalF(self, q_start,q_end,L):
        #Initialize matrix
        fe_local = np.zeros((6,1))
        #Evaluate local element stiffness matrix without any condensation
        f_local = self.LocalF_DSM(q_start,q_end,L)
        k_local = self.LocalK_DSM(L)
        #Apply static condensation
        alldofs = [0,1,2,3,4,5]
        retained_dofs = [0,1,2,3,4]
        condensed_dofs = np.setdiff1d(alldofs, retained_dofs)

        Krc = k_local[retained_dofs,:][:,condensed_dofs]
        Kcc = k_local[condensed_dofs,:][:,condensed_dofs]

        Fr = f_local[retained_dofs]
        Fc = f_local[condensed_dofs]
        
        fe_local_temp = Fr - Krc.dot(np.linalg.inv(Kcc)).dot(Fc)
        
        fe_local[retained_dofs,:] = fe_local_temp
    
        return fe_local