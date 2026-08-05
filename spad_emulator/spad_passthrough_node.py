#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class SpadEmulatorNode(Node):
    def __init__(self):
        super().__init__('spad_emulator_node')

        # Tunable parameters -- declared here so they can be changed at
        # launch time (or live via `ros2 param set`) without recompiling.
        self.declare_parameter('max_photons', 2000.0)
        self.declare_parameter('dead_time_ns', 20.0)
        self.declare_parameter('exposure_ns', 2000.0)  # 1ms, arbitrary placeholder

        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, '/spad/image_raw', self.callback, 10)
        self.pub = self.create_publisher(
            Image, '/spad/image_photon_realistic', 10)
        self.get_logger().info('SPAD emulator node started (Poisson + dead-time model active).')

    def callback(self, msg: Image):
        # Request mono8 directly -- cv_bridge/OpenCV handles the RGB->gray
        # conversion for us, since the real SPAD512 is monochrome.
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8').astype(np.float64)

        max_photons = self.get_parameter('max_photons').value
        dead_time_ns = self.get_parameter('dead_time_ns').value
        exposure_ns = self.get_parameter('exposure_ns').value

        # 1. Rendered brightness (0-255) -> "true" expected photon count per pixel
        lambda_true = (img / 255.0) * max_photons

        # 2. Dead-time compression (non-paralyzable model), applied as a rate
        #    over the exposure window, then converted back to a per-exposure count.
        true_rate = lambda_true / exposure_ns          # photons per ns
        dead_time_frac = true_rate * dead_time_ns       # unitless
        measured_rate = true_rate / (1.0 + dead_time_frac)
        lambda_measured = measured_rate * exposure_ns    # back to counts per exposure

        # 3. Poisson-sample the actual detected count using that compressed mean.
        #    This is the real noise-injection step -- everything above just
        #    computed what the mean *should* be.
        photon_counts = np.random.poisson(lambda_measured)

        # Scale back down to a viewable 8-bit range for display/output.
        # Normalize against the dead-time saturation ceiling (the max count
        # physically achievable at this exposure/dead-time combo), not max_photons --
        # those aren't the same thing once dead-time compression is significant.
        saturation_ceiling = min(max_photons, exposure_ns / dead_time_ns)
        out_img = np.clip(photon_counts / max(saturation_ceiling, 1.0) * 255.0, 0, 255).astype(np.uint8)

        out_msg = self.bridge.cv2_to_imgmsg(out_img, encoding='mono8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SpadEmulatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()