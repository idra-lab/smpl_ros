from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import numpy as np
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.actions import TimerAction
from scipy.spatial.transform import Rotation as R
def generate_launch_description():
    # Declare launch arguments for configurability
    rmw_zenoh_node = Node(
        package='rmw_zenoh_cpp',
        executable='rmw_zenohd',
        name='rmw_zenohd',
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.expanduser('~') + '/.rviz2/default.rviz']
    )

    camera_tf_broadcaster_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('easy_handeye2'), 'launch', 'publish.launch.py')
        )
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/SMPL_MALE.npz',
        description='Path to the SMPL model file (.npz)'
    )
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='world',
        description='Frame ID to which the SMPL model will be attached'
    )

    #   x: -2.773591995239258
    #   y: 0.8304675221443176
    #   z: -0.6756067872047424
    # orientation:
    #   x: 0.0005706624942831695
    #   y: 3.840220961137675e-05
    #   z: 0.0017058025114238262
    #   w: 0.9999984502792358

    transl_mocap_mrb = np.array([-2.773591995239258, 0.8304675221443176, -0.6756067872047424])
    quat_mocap_mrb = np.array([0.0005706624942831695, 3.840220961137675e-05, 0.0017058025114238262, 0.9999984502792358])

    # Normalizzazione (sempre buona pratica)
    quat_mocap_mrb = quat_mocap_mrb / np.linalg.norm(quat_mocap_mrb)

    mocap_T_mrb = np.eye(4)
    mocap_T_mrb[0:3, 0:3] = R.from_quat(quat_mocap_mrb).as_matrix()
    mocap_T_mrb[0:3, 3] = transl_mocap_mrb


    # =========================
    # 2) robot_base_mocap → robot_base_ros
    # =========================
    mrb_T_rrb = np.array([
        [1.0,  0.0,  0.0, 0],
        [0.0,  0.0,  1.0, 0],
        [0.0, -1.0,  0.0, 0],
        [0.0,  0.0,  0.0, 1]
    ])


    # =========================
    # 3) Composizione
    # =========================
    # (mocap → mrb) @ (mrb → rrb) = mocap → rrb

    mocap_T_rrb = mocap_T_mrb @ mrb_T_rrb


    # =========================
    # 4) Inversione (se serve)
    # =========================
    # vogliamo robot_base_ros → mocap

    rrb_T_mocap = np.linalg.inv(mocap_T_rrb)


    # =========================
    # 5) Estrazione per ROS
    # =========================

    translation = rrb_T_mocap[0:3, 3]
    quaternion = R.from_matrix(rrb_T_mocap[0:3, 0:3]).as_quat()


    # =========================
    # 6) Static TF publisher
    # =========================

    static_tf_broadcaster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_broadcaster',
        output='screen',
        arguments=[
            str(translation[0]), str(translation[1]), str(translation[2]),
            str(quaternion[0]), str(quaternion[1]), str(quaternion[2]), str(quaternion[3]),
            'world',   # parent
            'mocap'             # child
        ]
    )
        # Node configuration
    smpl_ros_viewer_node = Node(
        package='smpl_ros',
        executable='smpl_ros_viewer',
        name='smpl_ros_viewer',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'frame_id': LaunchConfiguration('frame_id')
        }]
    )
    activate_optitrack_node = ExecuteProcess(
        cmd=['ros2', 'lifecycle', 'set', '/mocap4r2_optitrack_driver_node', 'activate'],
        output='screen'
    )

    lifecycle_after_rmw = RegisterEventHandler(
        OnProcessStart(
            target_action=rmw_zenoh_node,
            on_start=[
                TimerAction(
                    period=2.0,
                    actions=[activate_optitrack_node]
                )
            ]
        )
    )
    return LaunchDescription([
        # rmw_zenoh_node,
        static_tf_broadcaster_node,
        camera_tf_broadcaster_launch,
        rviz_node,
        model_path_arg,
        frame_id_arg,
        smpl_ros_viewer_node,
        lifecycle_after_rmw
    ])