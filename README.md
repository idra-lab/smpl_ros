# 🌟 smpl_ros  
*A ROS 2 package for integrating SMPL with real-time Rviz visualization*  

---

## 🚀 Overview  
`smpl_ros` brings the **SMPL human body model** into the ROS 2 ecosystem with seamless **Rviz visualization support**.  
It’s designed to make working with 3D human meshes in robotics and visualization pipelines **simple, fast, and flexible**.  

---

## 📦 Dependencies  
To get started, make sure you have the following installed:  

- **ROS 2** (tested with **Humble**)  
- **[Open3D C++ library](https://github.com/isl-org/Open3D/releases)**  
  - Used for point cloud loading and downsampling  
  - *Alternatively, you can use PCL with minor code changes*  
- **CUDA** *(optional)* – Enables GPU acceleration  
- **[torchure_smplx](https://github.com/Hydran00/torchure_smplx)** – C++ SMPL Torch implementation  

---

## ⚙️ Installation  

1. Export required paths to `Open3D` and `torchure_smplx` CMake files
    ```
    export Open3D_DIR=~/path_to_your_open3d_build/lib/cmake/Open3D
    export torchure_smplx_DIR=~/path_to_your_torchure_smplx_build
    ```
2. (Optional) Configure CUDA  
    ```
    CUDA_VERSION=<your_cuda_version>
    export PATH="/usr/local/cuda-${CUDA_VERSION}/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:$LD_LIBRARY_PATH"
    ```
3. Clone the repository   
    ```
    cd ~/your_ros2_ws/src
    git clone git@github.com:Hydran00/smpl_ros.git
    ```
4. Build the workspace  
    ```
    cd ~/your_ros2_ws
    colcon build --symlink-install
    ```


