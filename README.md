## Documentation
This documentation outlines the structure and functioning of this project, designed for advanced image processing tasks using various machine learning models and libraries. The script combines object detection, segmentation, and inpainting capabilities to manipulate and enhance images according to specific user inputs.

### Script Overview

The primary goal of this script is to:
1. Detect objects in an image based on a text prompt.
2. Annotate these objects.
3. Segment the detected objects using a segmentation model.
4. Shift the segmented objects within the image.
5. Inpaint the original location of the objects to blend seamlessly with the background.

### Detailed Step-by-Step Explanation

#### 1. Importing Libraries
The script starts by importing necessary Python libraries and modules:
- argparse: For parsing command-line options.
- cv2 (OpenCV): For image processing.
- torch: For handling operations related to PyTorch tensors.
- numpy: For numerical operations on arrays.
- Various modules from custom libraries (groundingdino, supervision, segment_anything, diffusers) and PIL for handling specific model loading, predictions, and image manipulations.

#### 2. Object Detection and Annotation
- *Model Loading (load_model)*: Loads a pre-trained object detection model from specified configuration and weight files.
- *Image Loading (load_image)*: Reads an image from the disk and preprocesses it for the model.
- *Prediction (predict)*: The model predicts bounding boxes, logits (confidence scores), and phrases (labels) based on the given image and text prompt.
- *Annotation (annotate_)*: Annotates the detected objects on the image using the bounding box data, logits, and labels.
- *Dino model is leveraged for this task.

#### 3. Object Segmentation
- *Initialization (init_sam)*: Initializes a segmentation model from a registry based on the specified encoder version and checkpoint path.
- *Segmentation (segment)*: Applies the segmentation model to the detected objects, extracting and returning the highest confidence mask for each object.
- The famous SAM(Segment Anyhting Model)  by facebook is utilized for this task.

#### 4. Object Shifting
- *Shifting (shift_object)*: Now we have the object mask of the object specified by the user in the prompt. This function removes the segmented object in the image based on specified x and y offsets. This includes handling of image boundaries and repositioning of the object. This function returns an image containing original scene, and black  pixels on the segmented object. This black region is created to further remove any bias of orginal object in image in the process of inpainting. This also returns oobject region, which contains the shifted  object.

#### 5. Image Inpainting
- *Inpainting Setup (inpaint)*: Prepares and executes inpainting using a model from the diffusers library, specifically StableDiffusionInpaintPipeline. The script inpaints the area from where the object was moved, aiming to blend it naturally with the surrounding area. Inpinating is done on the whole image instead of a patch, as I observed better results by passing full image with segmented pixels black,and mask.

#### 6. Command-Line Interface Setup
- *Argument Parsing*: Sets up and parses command-line arguments for the image path, text prompt, x and y shifts, and output path.
- *Main Execution Flow*: Orchestrates the calling of the various functions based on the parsed arguments, managing the flow from detection to inpainting.

#### 7. Mask Bounding Box Calculation
- *Bounding Box Calculation (mask_bbox)*: Computes a bounding box around a binary mask, adding padding as specified. This function is crucial for defining the region to inpaint.

#### 8. Utility Functions
- *Bounding Box Conversion (_box_cxcywh_to_xyxy)*: Converts bounding boxes from center coordinates to corner coordinates format, which is necessary for certain operations and consistency across processing steps.

### Summary

The script effectively combines multiple advanced image processing techniques to manipulate images based on textual descriptions. It leverages deep learning models for object detection, segmentation, and inpainting, making it a powerful tool for image editing tasks. Each step is carefully designed to maintain the integrity and quality of the image, while dynamically adapting to user inputs.

## Installation
- * create new env:
 conda create -n move.

- * Setup DINO: Follow the steps here to install DINO:
https://github.com/IDEA-Research/DINO#installation

- *Setup SAM:Follow the steps here to install SAM:
https://github.com/facebookresearch/segment-anything#installation

- * download all dependencies specified in environment.yml
conda env export > environment.yml

## Usage
- * python get_masks.py --image_path <image path> --text_prompt <text_prompt> --op_path <op_path>
- This script takes the input scene and the text prompt
from the command line argument and outputs an image with a red mask on all pixels where
the object (denoted in the text prompt) was present.

 - * python translate_obj.py --image_path <image path> --text_prompt <text_prompt> --x <x> --y <y> --op_path <op_path>
 -This script changes the position of the segmented object using user prompts
specifying the number of pixels to shift in x (+x being the horizontal right direction) and y (+y
being the vertical up direction) directions.


