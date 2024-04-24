#Import dependencies and entities to set up FEA
import source.geometry as geometry
import source.model as model
import source.solvers as solvers
import numpy as np

####### Create geometry #######
#Mesh = geometry.mesh("Assignment_2023")
Mesh = geometry.mesh("Assignment_2024")
Mesh.getMesh()

FEMmodel = model.Model(Mesh)    	
FEMmodel.make()
#Visualize mesh
#FEMmodel.PlotDeformed()

####### Solve #######
#U = solvers.solveNumerical(FEMmodel) 
#FEMmodel.PlotDeformed()

freqs, eigens = solvers.solveModal(FEMmodel, print_results=True) 
FEMmodel.PlotDeformed(eigens[:,0])
print(ARK)