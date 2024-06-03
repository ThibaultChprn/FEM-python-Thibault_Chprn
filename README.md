# FEM-python-Thibault_Chprn

This repository contains a Python-based implementation of Finite Element Analysis to carry out structural analysis tasks. These include computing the deformed configuration of a system under a given excitation, calculating reaction and section forces, and visualizing the stresses along the structure.
The code is largely based on the Method of Finite Elements I course, taught at ETH Zurich by Prof. Chatzi.
The implementation includes truss and beam elements along with two-dimensional plane stress elements (triangles and quadrilaterals).

## Material repository:

https://polybox.ethz.ch/index.php/s/el24YlfFv2FGCYV

The background literature and additional material on ABAQUS and FEM theory are included in the polybox repository above. 

## Next Short-Term Steps:
- Improve summary graph/mind map and prepare for update progress meeting with Eleni
- Recap Lecture 7, 2D Elements and Feliba book, Chapter 9 (the Table). Also, check the PowerPoint slides on shearlocking [here](https://github.com/ThibaultChprn/FEM-python-Thibault_Chprn/blob/main/ShearLocking.pptx).
- Check the analysis [here](https://enterfea.com/what-are-the-types-of-elements-used-in-fea/) and [here](https://enterfea.com/why-is-a-triangular-element-stiffer/) on how to decide the type of element to use and how 2D elements compare. Also, check the second PowerPoint in 2D Elements [here](https://github.com/ThibaultChprn/FEM-python-Thibault_Chprn/blob/main/TRIQUADDemo.pptx).
- ABAQUS: Familiarize with existing tutorials and case studies
- ABAQUS: Structural analysis on example bridge structure. The system of interest is located here: [ABAQUS Bridge file](StructuralSystems/BridgeModel)
- Study Chapters 9-12 from Chopra's book and Chapter 31 from Felipa on dynamics. Focus on modal analysis and results interpretation.
## Next Long-Term Steps:
- Produce a structural analysis report on the bridge, including modal analysis in the form of a short progress-update presentation. 
- Compare ABAQUS results with Python template results (if possible)
