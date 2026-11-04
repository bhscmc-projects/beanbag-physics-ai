"""
sample_generator.py: Generates a high-quality simulated slow-motion video
using universal H.264 (yuv420p) codec so it plays natively in Google Chrome & Safari.
"""

import cv2
import numpy as np
import imageio


def generate_synthetic_toss_video(output_path, fps=120, duration=2.0):
    width, height = 640, 480
    total_frames = int(fps * duration)

    # Use imageio with libx264 and yuv420p for universal browser playback
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec='libx264',
        ffmpeg_params=['-pix_fmt', 'yuv420p']  # Required for Chrome/Safari HTML5 video
    )

    t = np.linspace(0, 1.8, total_frames)
    t_peak = 0.9
    max_h = 280.0  # height in pixels

    # Parabolic trajectory
    a = max_h / (t_peak ** 2)
    h_t = -a * (t - t_peak) ** 2 + max_h
    y_coords = 380 - h_t
    x_coords = 320 + 20 * np.sin(np.pi * t / t_peak)

    for i in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light background

        # Draw simulated lab desk
        cv2.rectangle(frame, (0, 420), (width, height), (180, 180, 180), -1)
        cv2.putText(frame, "Physics Lab 5.2 - Slow-motion Simulation", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

        cx = int(x_coords[i])
        cy = int(y_coords[i])

        # Red beanbag
        size = 20
        cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (30, 30, 220), -1)
        cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 150), 2)

        # Hand indicator
        if i < int(0.2 * fps) or i > int(1.6 * fps):
            cv2.ellipse(frame, (320, 410), (35, 15), 0, 0, 360, (190, 210, 240), -1)

        # Convert BGR (OpenCV) to RGB (for web video)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    writer.close()


if __name__ == "__main__":
    generate_synthetic_toss_video("test_toss.mp4")
    print("Generated test_toss.mp4 in browser-ready H.264 format.")