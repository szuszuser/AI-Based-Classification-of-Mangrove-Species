# **Mangrove Species Classification**



This project presents code and data for classifying mangrove species using machine learning (ML) and deep learning (DL).



#### **Machine Learning (ML)**

The ML package includes seven Python scripts, one model, and two raster images.

#### ***sample division.py***

Perform stratified KS sample partitioning on the input raw sample data (.xlsx). The partitioning results yield two tables: one for training data and one for test data.

#### ***Genetic\_algorithm.py*** 

#### ***Life.py***

#### ***Featire\_selection.py***

Combine the partitioned sample data (training data and test data) with the machine learning model to perform GA feature selection, ultimately outputting the results of each selection round and the optimal outcome.

#### ***FeatireSelectedResult.py***

Based on the selected features, obtain the model's accuracy results and perform parameter tuning.

#### ***Model.py***

Establish the final model based on the optimized parameters and save it.

#### ***Map.py***

Perform mapping on the original imagery based on the established model.

#### ***OriginalSamplePoint\_Partial.xlsx (Data)***

Sample Data Examples Provided.

#### ***RandomForest\_Optical\_SAR\_feature\_model (Model)***

Model file created

#### ***all\_selected\_feaure.dat (Image)***

Original image data for mapping

#### ***qiaodao\_Optical\_SAR\_RandomForest\_pixel.dat (Image)***

Classification Results Chart



## **Deep Learning (DL)**

The DL file contains three folders and two model files.

### **01\_Dataeset (File)**

Sample data for model training

#### ***iamge(file)***

Image folder for model training

#### ***mask(file)***

Mask folder for model training

### **02\_reprocess (File)**

Processing raw image data to generate a dataset suitable for model training and testing.

#### ***01\_padding.py***

Used to resize the original image to fit a specific cropping window.

#### ***02\_DataSubsetAndMosaic.py***

Crop the image using a specific cropping window and stitch the test results together.

#### ***03\_MaskSelect.py***

Filter the label data to exclude label images with all values set to zero.

#### ***04\_GetSameNameImage.py***

Filter images based on the label name.

#### ***05\_Projection.py***

Project the classified results to facilitate subsequent analysis.

#### ***SegNet173 (Model)***

Model Instance Based on SegNet

#### ***UNet200 (Model)***

Model Instance Based on UNet











