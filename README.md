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
- `util.py`: Customized math functions applied to this project.
- `score.py`:  Extract information to support the final scoring.


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

### 6. License
This is a commercial project, hence no Open Source Protocols here.
