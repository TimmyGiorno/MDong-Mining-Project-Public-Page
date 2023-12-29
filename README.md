# MDong-Mining-Project

**Note**: Chinese version of this `README.md` is available here.

### 1. Background
In the mine, we need to detect if the truck is full 
for each load. This problem can be solved 
by deep learning methods, but it also faces: 
1. the problem of generalizability with different car models and mineral types in different mines and different backgrounds; 
2. the fast iterative computation required in mines, so stronger interpretability and parameter tunability are needed.

Based on this, we developed a fast, robust, 
and generalizable intelligent detection system.


### 2. File Structure and Usage
- `predict.py`: Apply semantic segmentation to images.
- `train.py`: Train your own truck-mineral-background segmentation model.
- `registration.py`: Perform template-based registrations for moved images.
- `roughnessDetect.py`: Process curves and detect pits.
- `volumeDetect.py`: Detect Detects whether the truck is full of minerals from a volumetric point of view.
- `fullnessDetect.py`: From the position of the front and back of the mineral, detect whether the mineral is full or not.
- `utils.py`: Auxiliary mathematical functions.
- `visualization.py`: Auxiliary visualization functions.
- `score.py`: Rate the images.

### 3. Result Visualization



### 4. Contributor
- Bingchu Zhao (https://github.com/TimmyGiorno)
- **Email**: albernttimmy@outlook.com
- Han Lyu (https://github.com/ShimizuYoshiKazu)
- **Email**: 3567815517@qq.com
 
### 5. Timeline
- 03/27/2023: The draft codes were organized.
- 04/07/2023: The first version of registration codes were finished.
- 04/17/2023: The first version of segmentation codes were finished.
- 04/25/2023: The first version of feature extraction were finished.
- 05/14/2023: The fundamental framework were accomplished. 


### 6. License
This is a commercial project, hence no Open Source Protocols here.  
