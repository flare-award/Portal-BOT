"""Visual odometry and optical flow motion estimator for Portal-BOT."""

import math
from typing import Optional, Tuple
import cv2
import numpy as np

from portal_bot.config import VisionConfig


class VisualOdometry:
    """
    Estimates horizontal camera rotation (yaw) and linear motion magnitude
    from screen optical flow without dead-reckoning vertical pitch drift.
    """

    def __init__(self, config: VisionConfig):
        self.config = config
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.estimated_yaw: float = 0.0
        self.estimated_pitch: float = 0.0
        self.estimated_pos = [0.0, 0.0, 1.0]

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.estimated_yaw = 0.0
        self.estimated_pitch = 0.0
        self.estimated_pos = [0.0, 0.0, 1.0]

    def update(self, frame: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculates frame-to-frame optical flow.
        Returns: (flow_dx, flow_dy, motion_magnitude)
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0, 0.0

        flow_dx, flow_dy, mag = 0.0, 0.0, 0.0

        try:
            # Good features to track for sparse optical flow
            if self.prev_points is None or len(self.prev_points) < 30:
                self.prev_points = cv2.goodFeaturesToTrack(
                    self.prev_gray,
                    maxCorners=self.config.optical_flow_points,
                    qualityLevel=0.02,
                    minDistance=self.config.optical_flow_min_distance
                )

            if self.prev_points is not None and len(self.prev_points) > 10:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_points, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
                )

                if next_pts is not None and status is not None:
                    good_prev = self.prev_points[status == 1]
                    good_next = next_pts[status == 1]

                    if len(good_prev) > 8:
                        deltas = good_next - good_prev
                        # Robust median flow delta
                        med_dx = float(np.median(deltas[:, 0]))
                        med_dy = float(np.median(deltas[:, 1]))
                        
                        flow_dx = med_dx
                        flow_dy = med_dy
                        mag = math.hypot(flow_dx, flow_dy)

                        # Update estimated camera yaw (horizontal rotation)
                        # Moving scene right => camera turned left
                        yaw_step = - (flow_dx / float(w)) * 90.0
                        self.estimated_yaw += yaw_step
                        
                        # Pitch is kept stable at 0.0 to prevent runaway dead-reckoning drift
                        # Pitch adjustments are handled strictly via closed-loop visual aiming on targets
                        self.estimated_pitch = 0.0

                        self.prev_points = good_next.reshape(-1, 1, 2)
                    else:
                        self.prev_points = None
                else:
                    self.prev_points = None
            else:
                self.prev_points = None

        except Exception:
            self.prev_points = None

        self.prev_gray = gray
        return flow_dx, flow_dy, mag
