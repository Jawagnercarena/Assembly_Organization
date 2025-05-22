## Setup Part 1: Clone the Repository

Execute `git clone https://github.com/AllenInstitute/HebbsVision`, or otherwise clone https://github.com/AllenInstitute/HebbsVision into a directory of your choice.  If your cloning method supports git large files, this should take a while and download <X> gigs of files.

## Setup Part 2: Environment Setup
First, ensure that anaconda (conda) is installed on your system.
Then, to create the environment and install the conda and pip packages, at a command prompt execute `conda env create -n HebbsVision -f conda_files.yaml`
In the case of failure installing NeuroAnalysisTools, due to the lack of `requirements.txt`, you must create the environment first then download NeuroAnalysisTools from `https://github.com/zhuangjun1981/NeuroAnalysisTools`.  Once it exists in a local directory, you can install it locally from 

## Assembly Extraction and Hyperparameter Tuning (Optional):
Note that this step can take some time, so for those simply interested in reproducing our analysis of the extracted assemblies, the HebbsVision repository includes 
To extract the assemblies, first acquire the Similarity Graph Clustering implementation from https://github.com/janmoelter/sgc-assembly-detection.
Then, execute `python SGC.py preprocessing data_files/v1dd/functional/final/sessionM409828_13_dff.npy`.  Copy the resulting output files into `data_files/functional/final/`

## Figure 1

To recreate panel B of figure 1 (the only one displaying data from our analysis), open `AssemblyPaperFigures/FigureCode/Figure1/Figure1.ipynb` and execute all the cells in order.  
It will create files in the `AssemblyPaperFigures/FigureCode/Figure1/` directory.  Figure 1B uses `AssemblyPaperFigures/FigureCode/Figure1/3D_Scan_Images.png`, `AssemblyPaperFigures/FigureCode/Figure1/Example_Activity_Traces.png`, and `AssemblyPaperFigures/FigureCode/Figure1/raster_plot_red.png`.

## Figure 2

To recreate panels B, C, D, and the data for panel E of figure 2, open `AssemblyPaperFigures/FigureCode/Figure2/Figure2.ipynb` and execute all the cells in order.  
It will create files in the `AssemblyPaperFigures/FigureCode/Figure2/` directory.  Figure 2B uses `AssemblyPaperFigures/FigureCode/Figure2/Assemblies_Intersection_Upset_Plot.png`.  Figure 2C uses `AssemblyPaperFigures/FigureCode/Figure2/Assemblies_Plotted_In_Recording_Space_Same_Plot.png`.  Figure 2D uses `AssemblyPaperFigures/FigureCode/Figure2/Spatial_Distribution_of_Cells_by_Assembly.png`.  The data for Figure 2E are printed in the .ipynb notebook, following the cell where `compare_assemblies_spatial_distribution` is called.

## Figure 3

To recreate all panels in Figure 3, open `AssemblyPaperFigures/FigureCode/Figure3/Figure3.ipynb` and execute all the cells in order.  It will create files in the `AssemblyPaperFigures/FigureCode/Figure3/` directory

## Figure 4

To recreate all panels in Figure 4, open `AssemblyPaperFigures/FigureCode/Figure3/Figure4.ipynb` and execute all the cells in order.  It will create files in the `AssemblyPaperFigures/FigureCode/Figure4/` directory





Google Drive link for the larger data files, which are too big for github: https://drive.google.com/drive/folders/1jtjcpzZwZxY4yK1ycfz6BS5BVKjUJ_hc
