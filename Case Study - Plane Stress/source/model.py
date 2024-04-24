import source.element as element

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.patches as patches

class Model:
    def __init__(self, mesh, BCs, element_type="Tri"):
        self.material = Material()
        self.section = Section()
        self.mesh = mesh
        self.BCs = BCs
        if mesh['element_type'] == "Quad4":
            self.element = element.Quad4element(self.material, self.section)
        elif mesh['element_type'] == "Quad9":
            self.element = element.Quad9element(self.material, self.section)
        elif mesh['element_type'] == "Tri6":
            self.element = element.Tri6element(self.material, self.section)
        elif mesh['element_type'] == "Tri3":
            self.element = element.Tri3element(self.material, self.section)

    def make(self):

        dofs_per_node = 2
        nnodes = self.mesh["nodes"].shape[0]
        ndofs = nnodes * dofs_per_node
        self.equation = self.Equation(ndofs)

    def getSystemMatrices(self):
       
        self.assemble()

        K, f = self.equation.K, self.equation.f

        return K, f

    def assemble(self):

        #Initialization
        P, h, I = self.BCs["Amplitude"], self.section.A, self.section.I
        nodes = self.mesh['nodes']
        elements = self.mesh['elements']
        NumberOfElements = np.shape(elements)[0]

        N, Nr, Ns, points, weights = self.element.getShapeFunctions()

        for i in range(NumberOfElements):
                
            # 1. Get element nodes
            element_nodes = elements[i,:]
            # 2. Get nodal coordinates 
            coordinates = nodes[element_nodes-1,:]
            # 3. Evaluate element stiffness matrix
            Ke = self.element.surf_element_matrix(coordinates, self.material.C_plane_stress,Nr,Ns,points,weights)

            # 4. Assemble element stiffness matrix into global stiffness
            indexes = self.element.element_dofs(element_nodes)
            self.equation.K[np.ix_(indexes[:,0], indexes[:,0])] += Ke

        nedges = self.mesh['nedges']
        edges_r = self.mesh['edges_r']

        for i in range(nedges[0][0]):
                
                # 1. Get edge nodes
                nodes_edge = edges_r[i,:]
                # 2. Get nodal coordinates 
                coordinates = nodes[nodes_edge-1,:]
                # 3. Evaluate element stiffness matrix
                fe = self.element.edge_element_load(coordinates,P, h, I)     

                # 4. Assemble element stiffness matrix into global stiffness
                indexes = self.element.element_dofs(nodes_edge)
                self.equation.f[indexes] += fe

        self.ApplyBcs()

        

    def ApplyBcs(self):

        Kstiff, Fmodified = self.equation.K,self.equation.f

        #Nodes of the left edge
        nodes = self.mesh["nodes"]
        nodes_left = self.mesh['nodes_l']
        restrainedDofs = self.BCs['essential'][:,0].astype(int)
        self.restrainedDofs = restrainedDofs
        E, n = self.material.E, self.material.n        
        P, h, I = self.BCs["Amplitude"], self.section.A, self.section.I

        ##Evaluate analytical solution at nodes of the left 
        Uref = Model.Uanalytical(nodes[nodes_left-1,0],nodes[nodes_left-1,1],E, I, P, 10, n, h)
        Uref = np.squeeze(Uref, axis=1)
        Uref = np.squeeze(Uref, axis=2)

        #Convert to a vector
        Uref_vec = np.zeros((2*np.shape(nodes_left)[0]))
        Uref_vec[0::2]=Uref[0,:]
        Uref_vec[1::2]=Uref[1,:]

        #Modify load vector to account for constraints
        Fmodified = Fmodified - Kstiff[:,restrainedDofs].dot(Uref_vec)

        #Set rows and columns of the constraint dofs to zero
        Kstiff[:,restrainedDofs]=0
        Kstiff[restrainedDofs,:]=0

        #Set diagonal elements corresponding to constrained dofs to 1
        for k in range(len(restrainedDofs)):
            Kstiff[restrainedDofs[k],restrainedDofs[k]]=1

        #Set rhs equal to the imposed displacements
        Fmodified[restrainedDofs]=Uref_vec

        self.equation.K,self.equation.f = Kstiff, Fmodified


    def Uanalytical(x,y,E,I,P,L,n,h):
        Ua = -P/(6*E*I)*np.array([
            [-y*((6*L-3*x)*x + (2+n)*(y**2-h**2/4))],
            [3*n*(L-x)*y**2+(4+5*n)*h**2*x/4+(3*L-x)*x**2]
        ])
        
        return Ua


    # converts quad elements into tri elements
    def quads_to_tris(quads):
        tris = [[None for j in range(3)] for i in range(2*len(quads))]
        for i in range(len(quads)):
            j = 2*i
            n0 = quads[i][0]
            n1 = quads[i][1]
            n2 = quads[i][2]
            n3 = quads[i][3]
            tris[j][0] = n0
            tris[j][1] = n1
            tris[j][2] = n2
            tris[j + 1][0] = n2
            tris[j + 1][1] = n3
            tris[j + 1][2] = n0
        return tris

    # plots a finite element mesh
    def plot_fem_mesh(nodes_x, nodes_y, elements):
        for element in elements:
            x = [nodes_x[element[i]] for i in range(len(element))]
            y = [nodes_y[element[i]] for i in range(len(element))]
            plt.fill(x, y, edgecolor='black', fill=False)

    def plotNodal(self, nodalU=None):

        nodes = self.mesh["nodes"]
        elements = self.mesh['elements']-1

        if nodalU is None:
            nodalU = np.reshape(self.equation.u, (nodes.shape[0], -1))
            nodalU = nodalU[:,1]
        else: 
            nodalU = np.reshape(nodalU, (nodes.shape[0], 2))
            nodalU = nodalU[:,1]

        nodes_x,nodes_y = nodes[:,0], nodes[:,1]

        if self.mesh['element_type'] == "Quad4":
            rearrange = [0, 1, 2, 3]
            elements=elements[:,rearrange]
            ## convert all elements into triangles
            elements_all_tris = Model.quads_to_tris(elements)
        elif self.mesh['element_type'] == "Quad9":
            rearrange = [0, 1, 2, 3]
            elements=elements[:,rearrange]
            ## convert all elements into triangles
            elements_all_tris = Model.quads_to_tris(elements)
        elif self.mesh['element_type'] == "Tri6":
            rearrange = [0, 1, 2]
            elements=elements[:,rearrange]
            elements_all_tris=elements
        elif self.mesh['element_type'] == "Tri3":
            elements_all_tris=elements

        # create an unstructured triangular grid instance
        triangulation = tri.Triangulation(nodes_x, nodes_y, elements_all_tris)

        # plot the finite element mesh
        Model.plot_fem_mesh(nodes_x, nodes_y, elements_all_tris)

        # plot the contours
        plt.tricontourf(triangulation, nodalU)

        # show
        plt.colorbar()
        plt.axis('equal')
        plt.show()


    def plotDeformed(self, scale=1, Disps=None):
        nnodes = self.mesh["nodes"].shape[0]

        if Disps is None:
            displacements = np.reshape(self.equation.u, (nnodes, -1))
        else:
            displacements = np.reshape(Disps, (nnodes, -1))
            
        nodes = self.mesh["nodes"] 

        if self.mesh['element_type'] == "Quad4":
            rearrange = [0, 1, 2, 3]
        elif self.mesh['element_type'] == "Quad9":
            rearrange = [0, 4, 1, 5, 2, 6, 3, 7]
        elif self.mesh['element_type'] == "Tri6":
            rearrange = [0, 3, 1, 4, 2, 5]
        elif self.mesh['element_type'] == "Tri3":
            rearrange = [0, 1, 2]

        fig = plt.figure()
        ax = fig.gca()

        for element in self.mesh['elements']:
            element = element[rearrange]
            path = nodes[element-1,:]          
            ax.add_patch(patches.Polygon(path,facecolor="w", edgecolor="k"))
            ax = fig.gca()
            ax.axis('equal')

        for element in self.mesh['elements']:
            element = element[rearrange]
            path = nodes[element-1,:] + scale*displacements[element-1,:]           
            ax.add_patch(patches.Polygon(path,edgecolor="k"))
            ax = fig.gca()
            ax.axis('equal')

        plt.show()

    def plotCompareDeformed(self, Disps, scale=1):
        nnodes = self.mesh["nodes"].shape[0]

        displacements = np.reshape(self.equation.u, (nnodes, -1))
        displacementsC = np.reshape(Disps, (nnodes, -1))

        nodes = self.mesh["nodes"] 

        if self.mesh['element_type'] == "Quad4":
            rearrange = [0, 1, 2, 3]
        elif self.mesh['element_type'] == "Quad9":
            rearrange = [0, 4, 1, 5, 2, 6, 3, 7]
        elif self.mesh['element_type'] == "Tri6":
            rearrange = [0, 3, 1, 4, 2, 5]
        elif self.mesh['element_type'] == "Tri3":
            rearrange = [0, 1, 2]

        fig = plt.figure()
        ax = fig.gca()

        for element in self.mesh['elements']:
            element = element[rearrange]
            path = nodes[element-1,:] + scale*displacementsC[element-1,:]           
            ax.add_patch(patches.Polygon(path,edgecolor="k", facecolor="blue"))
            ax = fig.gca()
            ax.axis('equal')

        for element in self.mesh['elements']:
            element = element[rearrange]
            path = nodes[element-1,:] + scale*displacements[element-1,:]           
            ax.add_patch(patches.Polygon(path,edgecolor="k"))
            ax = fig.gca()
            ax.axis('equal')

        plt.show()


    def plotUndeformed(self):
        nodes = self.mesh["nodes"] 

        if self.mesh['element_type'] == "Quad4":
            rearrange = [0, 1, 2, 3]
        elif self.mesh['element_type'] == "Quad9":
            rearrange = [0, 4, 1, 5, 2, 6, 3, 7]
        elif self.mesh['element_type'] == "Tri6":
            rearrange = [0, 3, 1, 4, 2, 5]
        elif self.mesh['element_type'] == "Tri3":
            rearrange = [0, 1, 2]

        fig = plt.figure()
        ax = fig.gca()

        for element in self.mesh['elements']:
            element = element[rearrange]
            path = nodes[element-1,:]           
            ax.add_patch(patches.Polygon(path,edgecolor="k"))
            ax = fig.gca()
            ax.axis('equal')

        plt.show()


    class Equation:
        def __init__(self, ndofs):
            self.f = np.zeros(ndofs)
            self.u = np.zeros(ndofs)
            self.K = np.zeros((ndofs, ndofs))


class Material:
    def setProperties(self, E, n):
        self.E = E
        self.n = n
    
        self.C_plane_stress = E/(1-n**2)*np.array([
            [1, n, 0],
            [n, 1, 0],
            [0, 0, (1-n)/2]
            ])


class Section:
    def setProperties(self, A, I):
        self.A = A
        self.I = I
