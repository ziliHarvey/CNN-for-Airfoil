# CNN-based aerodynamics parameters prediction method
CNN for airfoil lift-to-drag-ratio prediction

This repository contains data, code, and results for implementing a airfoil lift-to-drag ratio prediction method based on Convolutional Neural Network. The network model can take into cnn model `the airfoil contour` and predict its `areodynamics parameters` such as lift-to-drag ratio.  
  
**/data/raw_data/foil_figure.rar:**   
&emsp;a file of all filled-in grayscale airfoil contour figures generated from coordinates txtx files downloaded from [UIUC Airfoil Data Site](https://m-selig.ae.illinois.edu/ads/coord_database.html).  
**/data/raw_data/csv.zip:**  
&emsp;a file of all samples' lift-to-drag ratio calculated by Xflr5 ![XFLR5-LOGO](http://www.xflr5.com/images/XFLR5_Logo.png)  
**/data/parsed_data/1_300.mat**  
&emsp;the above raw data is parsed into .mat file for loading  
**/source/raw_data_parsing.py**  
&emsp;This .py file shows steps of building a NIST36-alike organized dataset from raw data  
Please unzip 1_300.mat and modify directory and then run it. You can pass all data, which is a 4.2 GB 1_1550.mat file.  
**/source/CNN.py**  
&emsp;Run this file to load, train and print testset result
![](https://github.com/ziliHarvey/CNN--based-aerodynamics-parameters-prediction-method/raw/master/cnn.png)    

