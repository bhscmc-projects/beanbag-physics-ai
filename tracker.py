"""
tracker.py: 專業級紅色物體/西瓜追蹤模組
- 自動擴展橫向視野 (增加寬度比例，避免過於細長)
- 嚴格排除皮膚、木枱反光，優先鎖定空中目標
"""

import cv2
import numpy as np


class BeanbagTracker:
    def __init__(self, color_name="Red", min_radius=10, max_radius=220):
        self.min_radius = min_radius
        self.max_radius = max_radius

    def track_video(self, video_path, method="HSV"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 120.0

        ret, sample_frame = cap.read()
        if not ret:
            cap.release()
            return [], fps, {"x": np.array([]), "y": np.array([]), "radius": np.array([]), "valid": np.array([])}

        gray_sample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        col_means = gray_sample.mean(axis=0)
        row_means = gray_sample.mean(axis=1)
        valid_cols = np.where(col_means > 12)[0]
        valid_rows = np.where(row_means > 12)[0]

        if len(valid_cols) > 30 and len(valid_rows) > 30:
            raw_x1, raw_x2 = valid_cols[0], valid_cols[-1]
            raw_y1, raw_y2 = valid_rows[0], valid_rows[-1]
            
            # 關鍵改動：向左右兩側適度拓寬 22%，使畫面長寬比更闊、更自然
            w_box = raw_x2 - raw_x1
            pad_w = int(w_box * 0.22)
            crop_x1 = max(0, raw_x1 - pad_w)
            crop_x2 = min(sample_frame.shape[1], raw_x2 + pad_w)
            crop_y1, crop_y2 = raw_y1, raw_y2
        else:
            crop_x1, crop_x2 = 0, sample_frame.shape[1]
            crop_y1, crop_y2 = 0, sample_frame.shape[0]

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frames = []
        xs, ys, radii, valids = [], [], [], []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            h, w = cropped_frame.shape[:2]
            
            # 將基礎渲染高度設為 620px (原 800px)，配合寬度自然降低垂直長度
            scale = 620.0 / float(h)
            target_w, target_h = int(w * scale), 620
            frame_resized = cv2.resize(cropped_frame, (target_w, target_h))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            # 紅色雙重遮罩 (HSV + RGB 色差)
            hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 85, 60]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 85, 60]), np.array([180, 255, 255]))
            red_hsv_mask = cv2.bitwise_or(mask1, mask2)

            r_ch = frame_resized[:, :, 2].astype(np.int16)
            g_ch = frame_resized[:, :, 1].astype(np.int16)
            b_ch = frame_resized[:, :, 0].astype(np.int16)
            chroma_mask = ((r_ch - g_ch > 30) & (r_ch - b_ch > 25) & (r_ch > 95)).astype(np.uint8) * 255
            combined_mask = cv2.bitwise_and(red_hsv_mask, chroma_mask)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            best_candidate = None
            highest_score = -1.0

            for c in contours:
                area = cv2.contourArea(c)
                if area > 100:
                    ((cx, cy), r) = cv2.minEnclosingCircle(c)
                    if self.min_radius <= r <= self.max_radius:
                        altitude_bonus = (target_h - cy) / float(target_h)
                        score = area * (1.0 + altitude_bonus * 3.0)
                        if score > highest_score:
                            highest_score = score
                            best_candidate = (cx, cy, r)

            if best_candidate is not None:
                xs.append(best_candidate[0])
                ys.append(best_candidate[1])
                radii.append(best_candidate[2])
                valids.append(True)
            else:
                xs.append(0)
                ys.append(0)
                radii.append(0)
                valids.append(False)

        cap.release()

        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        radii = np.array(radii, dtype=np.float32)
        valids = np.array(valids, dtype=bool)

        valid_idx = np.where(valids)[0]
        if len(valid_idx) > 2:
            all_idx = np.arange(len(xs))
            xs = np.interp(all_idx, valid_idx, xs[valid_idx])
            ys = np.interp(all_idx, valid_idx, ys[valid_idx])
            radii = np.interp(all_idx, valid_idx, radii[valid_idx])

        return frames, fps, {
            "x": xs,
            "y": ys,
            "radius": radii,
            "valid": valids
        }