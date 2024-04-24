import numpy as np

def solveNumerical(system):

    system.make()

    Stiffness_K, fext = system.getSystemMatrices()
   
    Uf = np.linalg.solve(Stiffness_K, fext)

    #Reshape as matrix
    Uf = np.reshape(Uf,(int(len(Uf)/2),2))

    system.equation.u=Uf
    
    return Uf

def solveAnalytical(nodes, E,I,P,L,n,h):
    #Compute analytical solution
    Uexact = Uanalytical(nodes[:,0],nodes[:,1],E,I,P,L,n,h)
    Uexact = np.squeeze(Uexact, axis=1)

    return Uexact.T	

def Uanalytical(x,y,E,I,P,L,n,h):
    Ua = -P/(6*E*I)*np.array([
        [-y*((6*L-3*x)*x + (2+n)*(y**2-h**2/4))],
        [3*n*(L-x)*y**2+(4+5*n)*h**2*x/4+(3*L-x)*x**2]
    ])
        
    return Ua

