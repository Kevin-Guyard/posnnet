# POSNNET: POSitioning Neural Network Estimation Tool

## What is POSNNET

Many applications require **localizing an object in 2D or 3D** within environments of varying scale, without relying on additional infrastructure. A commonly preferred solution is to use a sensor composed of an **IMU** and a **GPS** chip. Data from the IMU and GPS are **fused through an algorithm** (for example, a Kalman Filter) to provide accurate estimates of the object's position, velocity, and orientation. While this solution yields satisfactory results when the GPS signal is strong, localization accuracy drops drastically when the **GPS signal is weak or unavailable**. 

👉 The POSNNET framework is **a solution designed to address this problem through postprocessing**. Leveraging advanced state-of-the-art artificial intelligence, **POSNNET can reconstruct the localization of the object during periods of weak or missing GPS signals**. The framework has been designed to be accessible to everyone, regardless of their experience with artificial intelligence. All you need is very basic knowledge of Python.

👉 The use of the POSNNET framework is quite simple. After setting up a POSNNET project, the framework will learn the dynamics of your sensor using the data you provide. Once this process is complete, you will be able to use the framework to reconstruct trajectories in situations where the GPS signal is weak or unavailable.

If you're interested in what's under the hood, please refer to the scientific publication that describes the framework: "POSNNET: POSitioning Neural Network Estimation Tool" by Guyard et al.

## Repository resources

- *User guide.ipynb*: A notebook that provides comprehensive information to understand and use the framework.

- *Use case example.ipynb*: A notebook that presents a use case example to illustrate how to utilize the framework in practice.

- *Framework validation.ipynb*: A notebook that presents the code and results used for the scientific validation of the framework.

- *project_framework_validation*: A folder containing the data of the framework validation project.

- *posnnet*: A folder containing the core code of the framework.

## Contact

👉 Do not hesitate to contact me if you require help or any information about the framework.

Kévin Guyard  
University of Geneva (Switzerland)  
Email: kevin.guyard@live.fr  
Linkedin: https://www.linkedin.com/in/kevin-guyard-8857aa191

## Contibutions

👉 Any contribution or bug report is welcome. Feel free to contact me.

## License

POSNNET © 2025 by Kévin Cédric Guyard is licensed under CC BY-SA 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-sa/4.0/