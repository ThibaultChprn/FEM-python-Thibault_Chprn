import source.element as element

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as patches

class Model:
    def __init__(self, mesh):
        self.mesh = mesh
        self.Elements=[]

        for k, v in enumerate(mesh.ElementProperties):
            element_type=v[-1]
            if element_type==0:
                self.Elements.append(element.BeamElement_Gauss(Material(v[0],v[1]), Section(v[2],v[3])))
            elif element_type==1:
                self.Elements.append(element.BeamElement_Gauss(Material(v[0],v[1]), Section(v[2],0)))           
            elif element_type==2 or element_type==3:
                self.Elements.append(element.BeamElement_Spring(Material(v[0],v[1]), Section(v[2],v[3])))  
            elif element_type==4:
                self.Elements.append(element.BeamElement_Condensed(Material(v[0],v[1]), Section(v[2],v[3]))) 
                
    
    def make(self):

        dofs_per_node = 3
        nnodes = self.mesh.nodes.shape[0]
        ndofs = nnodes * dofs_per_node
        self.equation = self.Equation(ndofs)

    def getFullSystemMatrices(self):
       
        self.assemble()

        K, f = self.equation.K, self.equation.f

        return K, f
    
    def getSystemMatrices(self):
       
        self.assemble()

        K, M ,f = self.equation.Kff,self.equation.Mff, self.equation.Ff

        return K, M, f

    def assemble(self):
        numberOfDofs = self.mesh.nodes.shape[0]*3
        #Matrix initialization
        K = np.zeros((numberOfDofs, numberOfDofs))
        M = np.zeros((numberOfDofs, numberOfDofs))
        F = np.zeros((numberOfDofs, 1))

        connectivity=self.mesh.connectivity.astype(int)
        crossSections=self.mesh.crossSections.astype(int)

        for e, (i, j) in enumerate(connectivity):
            
            # 1. Calculate length
            xi, yi = self.mesh.nodes[i, :] #Coordinates of the starting node
            xj, yj = self.mesh.nodes[j, :] #Coordinates of the ending node
            L = np.sqrt((xj-xi)**2+(yj-yi)**2)
            
            # 2. Calculate element's local stiffness matrix     
            qstart,qend = self.mesh.distributedLoads[e][0],self.mesh.distributedLoads[e][1]
            el = self.Elements[crossSections[e][0]]
            ke_local = el.LocalK(L)
            me_local = el.LocalM(L)
            fe_local = el.LocalF(qstart,qend,L)

            if L>0: #I need this check because my "virtual" spring truss has no length. Alternatively, you could have given a small length just to compute the rotation angle
                cos_theta, sin_theta = (xj-xi)/L, (yj-yi)/L
            else:
                cos_theta, sin_theta = 1,1
            
            R = np.array([
                [ cos_theta, sin_theta, 0,          0,         0, 0],
                [-sin_theta, cos_theta, 0,          0,         0, 0],
                [         0,         0, 1,          0,         0, 0],
                [         0,         0, 0,  cos_theta, sin_theta, 0],
                [         0,         0, 0, -sin_theta, cos_theta, 0],
                [         0,         0, 0,          0,         0, 1]
            ])

            # 4. Rotate element stiffness matrix to the global coordinate system
            
            ke_global = R.T.dot(ke_local).dot(R)
            me_global = R.T.dot(me_local).dot(R)
            fe_global = R.T.dot(fe_local)
        
        
            # 5. Assemble the element global stiffness to the system global stiffness
                
            K[3*i:3*i+3, 3*i:3*i+3] += ke_global[0:3, 0:3]
            K[3*i:3*i+3, 3*j:3*j+3] += ke_global[0:3, 3:6] 
            K[3*j:3*j+3, 3*i:3*i+3] += ke_global[3:6, 0:3] 
            K[3*j:3*j+3, 3*j:3*j+3] += ke_global[3:6, 3:6] 
            
            M[3*i:3*i+3, 3*i:3*i+3] += me_global[0:3, 0:3]
            M[3*i:3*i+3, 3*j:3*j+3] += me_global[0:3, 3:6]    
            M[3*j:3*j+3, 3*i:3*i+3] += me_global[3:6, 0:3]
            M[3*j:3*j+3, 3*j:3*j+3] += me_global[3:6, 3:6]

            F[3*i:3*i+3,0] += fe_global[0:3,0]
            F[3*j:3*j+3,0] += fe_global[3:6,0]

        for n, fx, fy, Rtheta in self.mesh.nodalLoads:
            F[3*int(n),0]+=fx
            F[3*int(n)+1,0]+=fy
            F[3*int(n)+2,0] += Rtheta

        self.equation.K,self.equation.M,self.equation.f = K, M, F       
        self.RotateSupport()
        self.ApplyBCs()


    def RotateSupport(self):
        Rot =  np.identity(self.mesh.nodes.shape[0]*3)
        rotatedBC = self.mesh.BCsRot

        for e, i in enumerate(rotatedBC):
            node,phi_left = i[0],i[1]
            node= node.astype(int)
            Rot[3*node,3*node]= np.cos(phi_left)
            Rot[3*node,3*node+1]= np.sin(phi_left)
            Rot[3*node+1,3*node]= -np.sin(phi_left)
            Rot[3*node+1,3*node+1]= np.cos(phi_left)

        self.equation.R = Rot
        Kfinal = Rot.T.dot(self.equation.K).dot(Rot)
        Mfinal = Rot.T.dot(self.equation.M).dot(Rot)
        Ffinal = Rot.T.dot(self.equation.f)

        self.equation.K, self.equation.M, self.equation.f = Kfinal, Mfinal, Ffinal
        

    def ApplyBCs(self):
        # All degree-of-freedom labels
        BCs = self.mesh.BCs
        allDofs = np.arange(0, self.mesh.nodes.shape[0]*3)

        restrainedDofs = []
        for i in range (0, BCs.shape[0]) :
            for j in range (1, BCs.shape[1]) :
                if BCs[i, j] == 1 :
                    dof = BCs[i,0]*3+j-1
                    restrainedDofs.append(dof)
        #Converting list to array
        restrainedDofs = np.array(restrainedDofs)
        self.mesh.restrainedDofs=restrainedDofs

        freeDofs = np.setdiff1d(allDofs, restrainedDofs)
        self.mesh.freeDofs=freeDofs

        Kff = self.equation.K[freeDofs, :][:, freeDofs]
        Kfr = self.equation.K[freeDofs, :][:, restrainedDofs]
        Krf = self.equation.K[restrainedDofs, :][:, freeDofs]
        Krr = self.equation.K[restrainedDofs, :][:, restrainedDofs]

        Ff = self.equation.f[freeDofs]

        Mff = self.equation.M[freeDofs, :][:, freeDofs]
        Mfr = self.equation.M[freeDofs, :][:, restrainedDofs]
        Mrf = self.equation.M[restrainedDofs, :][:, freeDofs]
        Mrr = self.equation.M[restrainedDofs, :][:, restrainedDofs]

        self.equation.Kff, self.equation.Mff, self.equation.Ff,self.equation.Krf = Kff, Mff, Ff, Krf

    def BackCalculate(self, print_output=True):
        numberOfElements = self.mesh.connectivity.shape[0]
        SectionForces = np.zeros((numberOfElements,6))
        crossSections=self.mesh.crossSections.astype(int)
        U = self.equation.u

        for e, (i, j) in enumerate(self.mesh.connectivity) :
            
            # 1. Calculate length    
            xi, yi = self.mesh.nodes[i, :]
            xj, yj = self.mesh.nodes[j, :]
            L = np.sqrt((xj-xi)**2+(yj-yi)**2)
            
            # 2. Calculate element's local stiffness matrix     
            qstart,qend = self.mesh.distributedLoads[e][0],self.mesh.distributedLoads[e][1]
            el = self.Elements[crossSections[e][0]]
            ke_local = el.LocalK(L)
            fe_local = el.LocalF(qstart,qend,L)
                
            # 3. Evaluate the rotation matrix
            if L>0: #I need this check because my "virtual" spring truss has no length. Alternatively, you could have given a small length just to compute the rotation angle
                cos_theta, sin_theta = (xj-xi)/L, (yj-yi)/L
            else:
                continue
            
            R = np.array([
                [ cos_theta, sin_theta, 0,          0,         0, 0],
                [-sin_theta, cos_theta, 0,          0,         0, 0],
                [         0,         0, 1,          0,         0, 0],
                [         0,         0, 0,  cos_theta, sin_theta, 0],
                [         0,         0, 0, -sin_theta, cos_theta, 0],
                [         0,         0, 0,          0,         0, 1]
            ]) 
                
            
            # Choose DOFs needed
            elementDofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]        
            Ue_global = U[elementDofs]
            
            # Rotate element global displacements to the local coordinate system
            Ue_local = R.dot(Ue_global)
            
            # Calculate local node forces
            Fe_local = ke_local.dot(Ue_local)
            
            # Save local node forces in matrix
            Ns, Ne = -(Fe_local[0][0]-fe_local[0][0])/1000, (Fe_local[3][0]-fe_local[3][0])/1000
            Vs, Ve = (Fe_local[1][0]-fe_local[1][0])/1000, -(Fe_local[4][0]-fe_local[4][0])/1000
            Ms, Me = (Fe_local[2][0]-fe_local[2][0])/1000, -(Fe_local[5][0]-fe_local[5][0])/1000
            SectionForces[e,:]=[Ns,Vs,Ms,Ne,Ve,Me]

            if print_output:            
                np.set_printoptions(precision=2)
                print('Element:',e)
                print('Normal force: Nstart=', np.round(Ns,2),'kN / Nend=', np.round(Ne,2),'kN')
                print('Shear force: Vstart=', np.round(Vs,2),'kN / Vend=', np.round(Ve,2),'kN')
                print('Moment: Mstart=', np.round(Ms,2),'kN / Mend=', np.round(Me,2),'kN')
                print("\n")

        self.equation.NQM = SectionForces

    def PlotDeformed(self, Up=None,scale_factor=10):
        plt.figure(figsize=(15,10))
        plt.axis('equal')

        if Up is None:
            Up=self.equation.u
        
        Uplot = np.reshape(Up, [self.mesh.nodes.shape[0], 3])

        # Loop over the rows of the connectivity matrix
        for element in self.mesh.connectivity:

            # Get the nodal displacements of each element
            x_coordinates = self.mesh.nodes[element.astype(int), 0]
            y_coordinates = self.mesh.nodes[element.astype(int), 1]

            # Plot the elements of the undeformed system
            plt.plot(x_coordinates, y_coordinates, 'k')

            # Get the nodal displacements of each element
            x_displacements = Uplot[element.astype(int), 0]
            y_displacements = Uplot[element.astype(int), 1]

            # Update the nodal coordinates of each element
            x_coordinates = x_coordinates+scale_factor*x_displacements
            y_coordinates = y_coordinates+scale_factor*y_displacements

            # Plot the elements of the deformed system
            plt.plot(x_coordinates, y_coordinates, 'r')

        plt.show()


    class Equation:
        def __init__(self, ndofs):
            self.f = np.zeros(ndofs)
            self.u = np.zeros(ndofs)
            self.K = np.zeros((ndofs, ndofs))
            self.M = np.zeros((ndofs, ndofs))

class Material:
    def __init__(self, E, rho=0):
        self.E = E
        self.rho = rho


class Section:
    def __init__(self, A, I):
        self.A = A
        self.I = I
