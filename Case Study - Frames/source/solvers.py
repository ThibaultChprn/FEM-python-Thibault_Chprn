import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs, eigsh

def solveNumerical(system, print_results=False):

    system.make()

    Stiffness_K, Mass_M, fext = system.getSystemMatrices()

    U = np.zeros((system.mesh.nodes.shape[0]*3, 1))
    Fext = np.zeros((system.mesh.nodes.shape[0]*3, 1))
    
    Uf = np.linalg.solve(Stiffness_K, fext)
    Fr = system.equation.Krf.dot(Uf)
    U[system.mesh.freeDofs] = Uf
    np.set_printoptions(precision=4)
    Fext[system.mesh.restrainedDofs] = Fr
    
    #Rotate results back
    Rot = system.equation.R
    U = Rot.dot(U)
    system.equation.u = U


    Unodal = np.reshape(U, [system.mesh.nodes.shape[0], 3])
    Fnodal = np.reshape(Fext, [system.mesh.nodes.shape[0], 3])

    if print_results:
        print('Nodal displacements (mm):\n')
        for ux, uy, Rtheta in Unodal:
            uxp, uyp, rp = "{:.2f}".format(ux*1000),"{:.2f}".format(uy*1000),"{:.2f}".format(-Rtheta*1000)
            print('Ux:'+str(uxp)+" mm, Uz:"+str(uyp)+" mm, Ry:"+str(rp)+" mrad")
        
        print('\nReaction forces (kN):\n')
        for ux, uy, Rtheta in Fnodal:
            uxp, uyp, rp = "{:.2f}".format(ux/1000),"{:.2f}".format(uy/1000),"{:.2f}".format(Rtheta/1000)
            print('Rx:'+str(uxp)+" mm, Rz:"+str(uyp)+" mm, My:"+str(rp)+" mrad")
        

    return Unodal, Fext

def solveModal(system, print_results=False):

    system.make()

    Stiffness_K, Mass_M, fext = system.getSystemMatrices()

    #scipy.sparse.linalg.eigs() => Check the documentation for the arguments of the function!
    eigvals_Gauss, eigvecs_Gauss = eigs(Stiffness_K, k=10, M=Mass_M, sigma=0)
    #Compute the natural frequencies using the eigenvalues
    frequencies_Gauss = np.sqrt(eigvals_Gauss.real)/(2*np.pi)
    #Sorth them and plot them properly
    idx = frequencies_Gauss.argsort()[::1]   
    frequencies_Gauss = frequencies_Gauss[idx]
    eigenVectors_Gauss = eigvecs_Gauss[:, idx].real

    eigenVectors = np.zeros((system.mesh.nodes.shape[0]*3,np.shape(eigenVectors_Gauss)[1]))
    eigenVectors[system.mesh.freeDofs,:]=eigenVectors_Gauss
    
    if print_results:
        np.set_printoptions(precision=2)
        print('Frequencies [Hz]:\n')
        print(frequencies_Gauss)

    return frequencies_Gauss, eigenVectors
