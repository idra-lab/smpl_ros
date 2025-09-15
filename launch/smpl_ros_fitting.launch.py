from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments for configurability
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', get_package_share_directory('smpl_ros') + '/rviz/fitting.rviz']
    )
    
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='path/to/SMPL_model.npz',
        description='Path to the SMPL model file (.npz)'
    )

    config_path_arg = DeclareLaunchArgument(
        'config_path',
        default_value='path/to/config.json',
        description='Path to the configuration file (.json)'
    )

    point_cloud_path_arg = DeclareLaunchArgument(
        'point_cloud_path',
        default_value='path/to/point_cloud.ply',
        description='Path to the point cloud file (.ply)'
    )

    # Node configuration
    fitting_smpl_node = Node(
        package='smpl_ros',
        executable='fitting_smpl',
        name='fitting_smpl_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'config_path': LaunchConfiguration('config_path'),
            'point_cloud_path': LaunchConfiguration('point_cloud_path')
        }]
    )
    delayed_fitting_smpl_node = TimerAction(
        period=2.0,
        actions=[fitting_smpl_node]
    )
    # Execute after 3 seconds delay to ensure RViz is ready

    return LaunchDescription([
        rviz_node,
        model_path_arg,
        config_path_arg,
        point_cloud_path_arg,
        delayed_fitting_smpl_node,
    ])