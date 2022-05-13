# Identifying fungal-infected apples
***
This is a project of multiple pattern recognition models identifying fresh apples and apples infected with different fungi, and we provide the raw and preprocessed data here, along with the corresponding codes.

#  Download code and datasets
***
run `git clone https://github.com/palenn/Identifying_fungal-infected_apples`.
The above command will download the data  code and all datasets automatically. You can also download your favorite dataset directly from our repository manually.

# Evaluate the dataset 
***
## Installation environment
- install [Matlab](https://ww2.mathworks.cn/products/matlab.html)2018b or newer version.
- setup the Conda virtual enviroment.
- * install [conda](https://www.anaconda.com) according to your computer system.
-  * start **conda prompt**
-  * run `conda create --name tf python=3.7`.
-  * run `conda activate tf `.
-  * run `pip install tensorflow`.
-  * run `pip install -r requirements.txt`
## Data preprocessing
using matlab to preprocessing data.
Note: Change the the correct path of  data according to your own situation.
### smooth filter
+  Switch the path to Identifying_fungal-infected_apples in Matlab, open the [Code/smooth](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Code/smooth) directory, and run [smooth_load.m](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/smooth/smooth_load.m) to get the smoothed data.
### Eliminate outliers
+ open [Code/sensors_eliminate_anomalous_data](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Code/sensors_eliminate_anomalous_data), and run [sensors_eliminate_anomalous_data.m](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/sensors_eliminate_anomalous_data/sensors_eliminate.m)
+ open [Code/Mahalanobis_eliminate_anomalous_data](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Code/Mahalanobis_eliminate_anomalous_data%00), and run [LoadData.m](9https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/Mahalanobis_eliminate_anomalous_data/LoadData.m)
### Extract feature parameters
+ open [Code/feature_parameters](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Code/feature_parameters), and run [feature_all.m](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/feature_parameters/feature_all.m).

***
using Pycharm to complete data dimensionality reduction and build pattern recognition model.
copy the Data and Code files just downloaded to the current project.
### Data dimensionality reduction
+ open [Code/data_dimensionality_reduction](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Code/data_dimensionality_reduction).
+ run [PCA.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/data_dimensionality_reduction/PCA.py) [FA.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/data_dimensionality_reduction/FA.py) and [LDA.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/data_dimensionality_reduction/LDA.py) will get data after dimensionality reduction by different methods.
### Building pattern recognition models
+ open [Code/pattern_recognition_model](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Code/pattern_recognition_model).
+ run [CNN.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/pattern_recognition_model/CNN.py)、[KNN.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/pattern_recognition_model/KNN.py)、[RF.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/pattern_recognition_model/RF.py)、[SVM.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/pattern_recognition_model/SVM.py)、[BPNN.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/pattern_recognition_model/BPNN.py)、[PSO_BPNN.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/pattern_recognition_model/PSO_BPNN.py)、[GWO_BPNN.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/pattern_recognition_model/GWO_BPNN.py) and [SSA_BPNN.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/pattern_recognition_model/SSA_BPNN.py) can get the performance of each model.
### Multi-pattern recognition platform
+ open [Code/Multi-algorithm_pattern_recognition_platform](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Code/Multi-algorithm_pattern_recognition_platform).
+ run [main.py](https://github.com/palenn/Identifying_fungal-infected_apples/blob/master/Code/Multi-algorithm_pattern_recognition_platform/main.py) will get Multi-algorithm_pattern_recognition_platform.

Notes:

 1. [Data/raw_data](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Data/raw_data) is  raw date.
 2. [Data/smoothed_data](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Data/smoothed_data) is the original data after smooth.
 3. [Data/sensors_eliminate](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Data/sensors_eliminate) is  the eliminated data by the difference between sensor No. 1 and sensor No. 5.
 4. [Data/feature_parameters_data](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Data/feature_parameters_data) is the data after feature extraction.
 5. [Data/eliminate_anomalous_data](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Data/eliminate_anomalous_data) is the data after removing outliers by Mahalanobis distance
 6. [Data/dimensionality_reduction_data ](https://github.com/palenn/Identifying_fungal-infected_apples/tree/master/Data/dimensionality_reduction_data)is the data after dimensionality reduction
