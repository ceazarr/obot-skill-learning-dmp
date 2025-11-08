# Skill Learning using Dynamic Movement Primitives (DMPs)

## Workflow

The process of recording and learning a new skill is divided into the following key stages:

1. Data Recording
- To capture a new skill demonstration, run the C++ executable:

    - ```SkillLearningDMP.cpp```

2. Post-Processing & Segmentation
- After recording, raw data must be segmented to isolate the skill execution. These scripts contain specific filters for this purpose.

- Run the appropriate script based on the skill being recorded:

    - ```post_processing_drawer.py```: For drawer-opening/closing skills.

    - ```post_processing_pickAndPlace.py```: For pick-and-place tasks.

**Note on Adaptation**: These scripts can be adapted for new skills. For a task like pushing (with no gripping), ```post_processing_pickAndPlace.py``` is a suitable starting point, as it includes a distance approximator from the wrist to the object. It may require minor modifications.

3. DMP Model Training
- Once the data is segmented and processed, you can train the DMP models.

- The following scripts contain the main training pipelines, including data filtering, outlier trimming, and DMP integration.

- Model Implementations:

    - ```ori_pos_dmp_lwr.py```: (Recommended) Trains an integrated position and orientation DMP using Locally Weighted Regression (LWR).

    - ```ori_pos_dmp_gmm.py```: Trains an integrated position and orientation DMP using Gaussian Mixture Models (GMM).

    - ```LinearRegression.py```: A basic Linear Regression model. **Note**: This file may require further development. 
    - 
LWR is the primary regression-based approach and can approximate Linear Regression if the kernel width is set to a very large value.

4. Generalization & Evaluation
- To test the generalization capabilities of a trained model, you can change the start and goal positions.

- Modify the ```DELTA_START``` and ```DELTA_GOAL``` variables within ```compare.py``` to evaluate performance of all the different models with new start/end-points.
**Note**: in this file the linear regression representation is done by taking the LWR and running it on one demo and keeping the kernel width as small as possible which is the same  as plain Linear Regression.

## Visualization Tools
This repository includes several scripts for visualizing different aspects of the data and models.

- ```grip_plotting_PickAndPlace.py```: Use this to visualize the segmentation thresholds and the distance between the wrist and the object for pick-and-place tasks.

- ```grip_plotting_Drawer.py```: Use this to visualize the segmentation thresholds for the drawer-opening task.

- ```compare.py```: Use this as a way to visualize all the models in the same graph to compare.

## Core Modules & Utilities
These Python modules contain the core functions imported by the main training and visualization scripts.

- ```cleaning_data.py```

    - Contains all data filtering and cleaning functions. This includes filters for filling gaps and applying constraints on joint movement.

- ```helper.py```

- Provides essential functions for DMP formulation and kinematics, including:

    - ```smooth_positions```
    - 
    - ```kinematics (velocity and acceleration)```
    - 
    - ```forcing_target (calculating the DMP forcing term)```
    - 
    - ```canonical_phase```
    - 
    - ```calculate_hand_orientation```

- **Note**: A function for smoothing orientation data is included but was not fully validated. The primary models currently use raw, unsmoothed orientation data.

- ```orientation_DMP.py```

    - Contains all necessary functions for quaternion mathematics (e.g., multiplication, logarithm, exponential map) required for the orientation DMP.

- ```ori_visual.py```

    - Provides helper functions for plotting and visualizing orientation data.

- ```visual_gmm.py```

- Despite the name, this module contains general-purpose functions for visualizing positional movement in 2D and 3D. It is used by both the LWR and GMM models.