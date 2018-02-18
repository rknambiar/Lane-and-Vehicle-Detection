# Lane Detection

In this project we aim to do a Lane and Vehicle detection pipeline to mimic Lane Departure Warning systems used in Self Driving Cars 

## Getting Started

The code is written in Python 3.6. OpenCV and [Tensorflow Object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) has been used for lane detection and vehicle detection part of the pipeline respectively. 

The python file can be implemented without any additional dependencies. For implementing the Vehicle detection part, you must download the TF Object Detection API and copy the python notebook along with the Videos Folder. 

The implementation pipeline is explained below. 

## Lane Detection Pipeline

In this pipeline, the idea is to detect the edges of the lanes after color thresholding, then to apply RANSAC for robust estimation.

### Step 1 - Color Thresholding and Denoise the image

After applying the masks for yellow and white colors, we get the following output

Yellow Mask - 

![Yellow](https://github.com/rohit517/CVision-RoadLane/blob/master/Images/Screenshot%20(175).png)

White Mask - 

![White](https://github.com/rohit517/CVision-RoadLane/blob/master/Images/Screenshot%20(176).png) 

### Step 2 - ROI extraction and Edge Detection 

THe output of the combined masks is passed through an edge detector. The region of interest is extracted as we only want to focus on the bottom half of the image where we expect to find the lanes.

Combined Masks and ROI - 

![Combined](https://github.com/rohit517/CVision-RoadLane/blob/master/Images/Screenshot%20(172).png)

Edge Detection - 

![Edge](https://github.com/rohit517/CVision-RoadLane/blob/master/Images/Screenshot%20(174).png)

### Step 3 - Hough Transform and RANSAC

To improve the robustness and improve the accuracy of lane detection, RANSAC was used on the points obtained after Hough transform step.

Lane Overlay with RANSAC - 

![Ransac](https://github.com/rohit517/CVision-RoadLane/blob/master/Images/Screenshot%20(173).png)

![Mask](https://github.com/rohit517/CVision-RoadLane/blob/master/Images/Screenshot%20(177).png)

### Step 4 - TF object Det Api integration

Finally, the Tensorflow Object detection API was integrated with Lane detection.

Lane and Vehicle Detection - 

![LVD](https://github.com/rohit517/CVision-RoadLane/blob/master/Images/Screenshot%20(179).png)

The final result after implementing is shown below. [Link to Video](https://www.youtube.com/watch?v=7Hx7mQV6f-c)

## Author

* **Rohitkrishna Namnbiar**  - [rohit517](https://github.com/rohit517)

## Acknowledgments

Lane Video was used from the Udacity Self Driving course for development.  





