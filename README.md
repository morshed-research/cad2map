# Indoor Mapmaking: From Map Drawing to Navigable Representations

This repository addresses the problem of creating a map representation of indoor spaces from indoor floorplans. It implements a multi-stage pipeline which takes a PNG image of an indoor floorplan (originally a CAD dwg file which has been exported to a PNG image) as input and attempts to output a graph structure that reasonably represents the indoor floorplan. This implementation was originally created as part of Huda Baig's Undergraduate Senior Thesis at Carnegie Mellon University in Qatar and then continued by a separate research team at Carnegie Mellon University in Qatar as part of a larger indoor navigation research project.

## Current Status
The project obtains all area and connectivity nodes by reading added text labels in floor plan images. It relies on splitting connective areas into multiple labels. Door nodes are then inferred with fluctuating accuracy through a door detection model. Edges are inferred with different closeness metrics with varying accuracy.

## Dependencies

### Python Version
Requires Python 3.10.2. A virtual environment is recommended for the set-up.

### Packages
To install all library dependencies, on a system with Python installed, run the following from the repository's root directory:
```
sh install/setup.sh
```

### Model Weights
We use model weights from a folder named 'model_weights' under the root repository directory. You can set up this folder and download the three necessary weight files using the provided script.
Run the following from the repository's root directory:
```
sh install/weights.sh
```

## Models Used
The pipeline implementation uses three external models. First, it uses the [CRAFT](https://github.com/clovaai/CRAFT-pytorch) text bounding box detection model to extract text bounding boxes from PNG image floorplans. Second, it uses the [Deep Text Recognition](https://github.com/clovaai/deep-text-recognition-benchmark) model to infer text labels from the PNG image floorplans. Finally, it uses a door bounding box detection model that we trained at [Door Detection Repository](https://github.com/morshed-research/Door_Detection_Model) to output door bounding boxes from PNG image floorplans. 

## Repository File Structure
### 'utils'
Folder containing base object classes and measure functions
- connective.py: contains class for identifying connective nodes globally
- distance.py: contains functions for computing and making use of distances between node coordinates
- graph.py: contains node and graph object classes

### 'slicing'
Folder containing relevant object classes and functions to slice an image into several subset images
- dataset.py: contains object classes needed to process PNG images for slicing into several subset images
- image_slice.py: contains functions to process a PNG image and save it as several subset images of similar size

### 'results'
Folder containing all intermediary and final results from pipeline development
- 'average': initial test on creating edges based on average distance threshold
- 'doors': initial tests on accumulating door edges with different converging methods
- 'euclidean': initial test on using fixed euclidean distance thresholds to create edges
- 'evaluation': final evaluation results for different parameters
- 'inference': results from identifying text bounding boxes on different floorplan images
- 'json': resulting JSONs representing the graph structure for final evaluation results
- 'pipeline': full pipeline at intermediary stages of development
- 'radial': initial tests on creating edges based on a fixed number of connections to the nearest nodes
- 'splicing': initial tests on splicing floorplan images
- 'testbeds': initial tests of loading ground truth JSONs

### 'linking'
Folder containing functions to create edges based on door nodes
- edges.py: contains a function to connect area nodes to the closest door nodes

### 'install' 
Folder containing all scripts and files needed to install dependent weights and libraries
- requirements.txt: *most* required Python packages
- setup.sh: script to install required python packages
- weights.sh: script to install needed model weights

### 'extraction' 
Folder containing code needed to extract all areas and doors from a PNG image
- 'door_model': code to extract door bounding boxes
- 'label_model': code to extract text bounding boxes
- area.py: code to create area nodes
- door.py: code to create door nodes
- merge.py: functions that check conditions for merging nodes that represent the same location and perform this merge when needed

### 'eval'
Folder containing objects and functions used for evaluating pipeline output
- metrics.py: contains the object to perform shortest path similarity evaluation
- testbed.py: contains functions to load output or ground truth graphs from JSON format

### 'data'
Folder containing test input data, image slices and ground truth files

### 'completion'
Folder for handling missing edge cases after initial convergence through doors
- edges.py: contains functions to create edges between doors and connective areas, and between connective areas themselves

## Run the Code
### Pipeline Execution 
To run the pipeline on a PNG image floorplan, execute the following in a terminal:
```
python main.py <path-to-image>
```

Note that the test image should be stored under the data folder, and the data folder should not be included as part of the path.

### Evaluation Execution
To run evaluation on a PNG image floorplan, execute the following in a terminal:
```
python main.py <path-to-image-json> <path-to-ground-truth-json> <path-to-original-image>
```
