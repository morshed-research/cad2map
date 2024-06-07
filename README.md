# Indoor Mapmaking: From Map Drawing to Navigable Representations

This repository addresses the problem of creating a map representation of indoor spaces from indoor floorplans. It implements a multi-stage pipeline which takes a PNG image of an indoor floorplan (originally a CAD dwg file which has been exported to a PNG image) as input and attempts to output a graph structure that reasonably represents the indoor floorplan. This implementation was originally created as part of Huda Baig's Undergraduate Senior Thesis at Carnegie Mellon University in Qatar and then continued by a separate research team at Carnegie Mellon University in Qatar as part of a larger indoor navigation research project.

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

## Run the Code
