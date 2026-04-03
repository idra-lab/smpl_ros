from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
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

    return LaunchDescription([
        # static_tf_broadcaster_node,
        camera_tf_broadcaster_launch,
        rmw_zenoh_node,
        rviz_node,
        model_path_arg,
        frame_id_arg,
        smpl_ros_viewer_node,
    ])