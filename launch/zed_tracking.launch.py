from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    static_tf_broadcaster_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_broadcaster",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "map", "map1"],
    )
    # Declare launch arguments for configurability
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", get_package_share_directory("smpl_ros") + "/rviz/vis.rviz"],
    )

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="path/to/SMPL_model.npz",
        description="Path to the SMPL model file (.npz)",
    )
    frame_id_arg = DeclareLaunchArgument(
        "frame_id", default_value="map", description="Frame ID for the SMPL model"
    )

    # Node configuration
    smpl_ros_viewer_node = Node(
        package="smpl_ros",
        executable="smpl_ros_viewer",
        name="smpl_ros_viewer",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "frame_id": LaunchConfiguration("frame_id"),
            }
        ],
    )

    #   node->declare_parameter<std::string>("calibration_file",
    #                                        "/home/nardi/smpl_ros/zed_calib3.json");
    #   node->declare_parameter<std::string>("yolo_model_path",
    #                                        "/home/nardi/smpl_ros/yolov8x-seg.onnx");
    #   auto tf_static_broadcaster_ =
    #       std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
    #   // used only with fusion API
    #   node->declare_parameter<int>("max_width", 1280);
    #   node->declare_parameter<int>("max_height", 720);
    #   node->declare_parameter<bool>("publish_point_cloud", true);
    #   node->declare_parameter<bool>("publish_image", false);
    #   node->declare_parameter<std::string>("point_cloud_output_file",
    #                                        "human_cloud.ply");
    #   node->declare_parameter<std::string>("smpl_params_file", "");

    zed_tracking_node = Node(
        package="smpl_ros",
        executable="zed_smpl_tracking",
        name="zed_smpl_tracking",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "frame_id": LaunchConfiguration("frame_id"),
            }
        ],
    )

    return LaunchDescription(
        [
            static_tf_broadcaster_node,
            rviz_node,
            model_path_arg,
            frame_id_arg,
            smpl_ros_viewer_node,
        ]
    )
