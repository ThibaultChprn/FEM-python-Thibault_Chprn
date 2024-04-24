#Import dependencies and entities to set up FEA
import source.geometry as mesh
import source.model as model
import source.solvers as solvers
import numpy as np

####### Create geometry #######
Mesh = mesh()
Mesh.getMesh("Assignment_2023")

####### Create "model" #######
#Instantiate model from respective class
FEMmodel = model.Model(Mesh)    	

#Visualize mesh
FEMmodel.plotDeformed()

####### Solve #######
U = solvers.solveNumerical(FEMmodel) 
FEMmodel.plotDeformed(scale=0.1)

print(ARK)