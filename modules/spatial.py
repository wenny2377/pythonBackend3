import torch
import cv2
import numpy as np
from config import Config

class SpatialReasoning:
    def __init__(self):
        self.device = Config.DEVICE
        self.model_type = Config.MIDAS_MODEL_TYPE
        
        print(f"--- Loading MiDaS depth model ({self.model_type}) ---")
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        print("✅ MiDaS model loaded successfully")

    def estimate_coordinate(self, cv2_img, robot_pos, robot_yaw, fov):
        """
        Input image and robot pose, precisely output estimated user coordinates (x, z)
        """
        # 1. Image preprocessing and depth inference
        input_batch = self.transform(cv2_img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=cv2_img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # 2. Extract depth information
        h, w = prediction.shape
        # For robustness, take the average of the central 10x10 region to avoid single-point noise
        center_zone = prediction[h//2-5:h//2+5, w//2-5:w//2+5]
        center_depth_raw = center_zone.mean().item()

        # 3. Depth calibration and safety bounds (Crucial Fix)
        # If MiDaS outputs 0, it usually means nothing was detected.
        # Force it to a reasonable indoor distance.
        if center_depth_raw < 0.1:
            estimated_distance = 2.5  # Default 2.5 meters
        else:
            # Use a more stable mapping method
            # Assume indoor MiDaS center_depth_raw ranges roughly between 100~500
            # Tune DEPTH_SCALE according to your scene scale (suggest starting from 0.1 ~ 1.0)
            estimated_distance = (500.0 / (center_depth_raw + 1.0)) * Config.DEPTH_SCALE

        # Clamp distance: indoor targets are unlikely beyond 5 meters
        estimated_distance = max(0.5, min(estimated_distance, 5.0))

        # 4. Inverse projection to world coordinates
        # Unity convention: Yaw 0 = +Z axis, 90 = +X axis (clockwise)
        yaw_rad = np.radians(robot_yaw)
        
        # Mathematical formulas:
        # Target_X = Robot_X + Distance * sin(Yaw)
        # Target_Z = Robot_Z + Distance * cos(Yaw)
        target_x = robot_pos['x'] + estimated_distance * np.sin(yaw_rad)
        target_z = robot_pos['z'] + estimated_distance * np.cos(yaw_rad)

        print(f"✅ Spatial reasoning : depth={center_depth_raw:.2f}, distance={estimated_distance:.2f}m")
        print(f"   Estimated point: ({target_x:.2f}, {target_z:.2f})")

        return {"x": float(target_x), "z": float(target_z)}