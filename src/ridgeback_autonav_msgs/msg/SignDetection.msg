# A confirmed (or candidate) room-number sign localized in the map frame.
std_msgs/Header header          # stamp + frame_id (= "map")
string text                     # normalized room number, e.g. "206"
geometry_msgs/Point position    # map-frame position of the sign
float32 confidence              # latest recognition confidence [0..1]
int32 observations              # number of consistent observations
bool confirmed                  # passed multi-observation confirmation
