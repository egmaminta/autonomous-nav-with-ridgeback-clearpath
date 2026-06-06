# Mission FSM status, published periodically by mission_node.
std_msgs/Header header
string task                  # original task string
string target                # parsed target room
string state                 # current FSM state
string result                # "" while running, else SUCCESS | FAILED_NOT_FOUND | ABORTED
float32 elapsed_s
geometry_msgs/Pose home_pose # captured start pose (map frame)
