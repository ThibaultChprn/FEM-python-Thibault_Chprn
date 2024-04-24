#Import dependencies and entities to set up FEA
import source.geometry as geometry
import source.model as model
import source.solvers as solvers
import numpy as np

####### Create geometry #######
Mesh = geometry.mesh("Assignment_2023") #-> OK!
#Mesh = geometry.mesh("Assignment_2024") #-> OK!
#Mesh = geometry.mesh("PlaneFrameDemo")  #-> OK!
#Mesh = geometry.mesh("Assignment_2022") #-> OK!
Mesh.getMesh()

FEMmodel = model.Model(Mesh)    	
FEMmodel.make()
#Visualize mesh
#FEMmodel.PlotDeformed()

####### Solve #######
U = solvers.solveNumerical(FEMmodel, True) 
FEMmodel.PlotDeformed()

#freqs, eigens = solvers.solveModal(FEMmodel, print_results=True) 
#FEMmodel.PlotDeformed(eigens[:,4])
print(ARK)