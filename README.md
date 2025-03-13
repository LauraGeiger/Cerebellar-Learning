[![GitHub pull-requests](https://img.shields.io/github/issues-pr/LauraGeiger/Cerebellar-Learning.svg?style=plastic)](https://github.com/LauraGeiger/Cerebellar-Learning/pulls)
[![GitHub issues](https://img.shields.io/github/issues/LauraGeiger/Cerebellar-Learning.svg?style=plastic)](https://github.com/LauraGeiger/Cerebellar-Learning/issues)

[![GitHub stars](https://img.shields.io/github/stars/LauraGeiger/Cerebellar-Learning.svg?label=Stars&style=social)](https://github.com/LauraGeiger/Cerebellar-Learning/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/LauraGeiger/Cerebellar-Learning.svg?label=Watch&style=social)](https://github.com/LauraGeiger/Cerebellar-Learning/watchers)

# Cerebellar-Learning
This project implements a biologically inspired, adaptive robot control system for a soft hand exoskeleton.
The system, modeled after the cerebellum, uses spiking neurons to regulate air pressure and inflation timing for the thumb and index finger, enabling real-time grasping. 
Its effectiveness is demonstrated in both simulated and physical environments.

The model is implemented using the **NEURON** simulation platform.

## Installation
To set up the project on your local machine, follow these steps:  
1. Clone the repository:
    ```
    git clone https://github.com/LauraGeiger/Cerebellar-Learning.git
    cd Cerebellar-Learning
    ```
2. Optional: Create a virtual environment
    ```
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate     # On Windows
    ```
3. Install the following dependencies:
* Python **>= 3.7** (**Note:** Tested with Python **3.10.16**)
* `numpy`
* `matplotlib`
* `pymata-aio`
Install dependencies using:
    ```
    pip install numpy matplotlib pymata-aio
    ```
4. Ensure **NEURON** is installed: [NEURON Installation Guide](https://www.neuron.yale.edu/neuron/download).

## Usage
### Simulation Only
1. Run `CerebellarLearningModel.py`:
    ```
    python .\CerebellarLearningModel.py
    ```
2. Modify settings using the GUI.

### With HW Control
1. Connect to the **Soft Robotics Edu. Toolkit** designed by John Nassour.
2. Upload the `StandardFirmata.ino` file to the **Arduino**.
3. Optional: Run `Arduino_SoftRoboticsEduToolkit.py` to check toolkit functionality.
    **Note:** Change port (e.g. port = "COM8") to match your computer.
    ```
    python .\Arduino_HelloWorld.py
    ```
4. Run `CerebellarLearningModel.py`.
    **Note:** Change port (e.g. port = "COM8") to match your computer.
    ```
    python .\CerebellarLearningModel.py
    ```
5. Modify settings using the GUI.

## Authors
* **Laura Geiger** → [git](https://github.com/LauraGeiger)