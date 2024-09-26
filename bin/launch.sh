SCRIPT_DIR=$(dirname "$(realpath "$0")")
DEV_CONTAINER_PATH=$(realpath "$SCRIPT_DIR/../dev_container")
$DEV_CONTAINER_PATH launch --ssh_x11 --img holohub:v2.3.0_h264_qt