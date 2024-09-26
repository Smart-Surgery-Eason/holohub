working_dir=$(dirname $(dirname $(realpath $0)))
$working_dir/dev_container launch --ssh_x11 --img holohub:v2.3.0_h264_qt