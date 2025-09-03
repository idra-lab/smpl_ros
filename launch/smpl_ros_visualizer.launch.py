from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments for configurability
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', get_package_share_directory('smpl_ros') + '/rviz/vis.rviz']
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='path/to/SMPL_model.npz',
        description='Path to the SMPL model file (.npz)'
    )


    # Node configuration
    smpl_ros_viewer_node = Node(
        package='smpl_ros',
        executable='smpl_ros_viewer',
        name='smpl_ros_viewer',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
        }]
    )

    return LaunchDescription([
        rviz_node,
        model_path_arg,
        smpl_ros_viewer_node,
    ])