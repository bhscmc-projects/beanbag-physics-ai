"""
physics.py: 專為中一科學設計的運動與能量計算模組
- 全面統一使用科學術語「速率 (Speed)」
- 100% 根據影片真實幀座標計算真實勢能與動能
"""

import numpy as np
from scipy.signal import savgol_filter


def smooth_and_interpolate(raw_y, valid_mask, window_length=15, polyorder=2):
    n = len(raw_y)
    indices = np.arange(n)
    
    if np.sum(valid_mask) < 5:
        return raw_y

    interp_y = np.interp(indices, indices[valid_mask], raw_y[valid_mask])
    
    if window_length >= n:
        window_length = n - 1 if (n - 1) % 2 != 0 else n - 2
    if window_length <= polyorder:
        window_length = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3

    try:
        smoothed_y = savgol_filter(interp_y, window_length=window_length, polyorder=polyorder)
    except Exception:
        smoothed_y = interp_y

    return smoothed_y


def calculate_real_frame_physics(smoothed_y, valid_mask, fps):
    n = len(smoothed_y)
    dt = 1.0 / fps

    # 1. 尋找真實最高點 (Apex: 像素 Y 最小處)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 5:
        apex_f = n // 2
    else:
        apex_f = int(valid_idx[np.argmin(smoothed_y[valid_idx])])

    # 2. 手部基準高度 (手持西瓜時的 Y 坐標基準)
    baseline_y = np.percentile(smoothed_y, 90)
    apex_y = smoothed_y[apex_f]
    max_height_px = max(10.0, baseline_y - apex_y)

    # 3. 計算每一幀真實高度 h
    real_h = np.maximum(0.0, baseline_y - smoothed_y)
    ep_real = np.clip((real_h / max_height_px) * 100.0, 0.0, 100.0)

    # 4. 計算每一幀真實垂直速率 (純量大小 Speed, 取絕對值)
    real_v = np.gradient(real_h, dt)
    real_speed = np.abs(real_v)

    # 5. 精確自動判定拋出點與接住點
    threshold_h = 8.0
    flight_indices = np.where(ep_real > threshold_h)[0]

    if len(flight_indices) > 0:
        start_f = max(0, flight_indices[0] - 2)
        end_f = min(n - 1, flight_indices[-1] + 2)
    else:
        start_f = max(0, apex_f - 20)
        end_f = min(n - 1, apex_f + 20)

    # 6. 計算真實動能 Ek (與速率平方成正比)
    flight_speeds = real_speed[start_f:end_f + 1]
    max_speed = np.percentile(flight_speeds, 95) if len(flight_speeds) > 0 else 1.0
    if max_speed <= 0:
        max_speed = 1.0

    ek_raw = ((real_speed / max_speed) ** 2) * 100.0
    ek_real = np.clip(ek_raw, 0.0, 100.0)

    # 拋出前與接住後，物體靜止在手，動能與勢能歸零
    for f in range(n):
        if f < start_f or f > end_f:
            ep_real[f] = 0.0
            ek_real[f] = 0.0

    # 最高點校準：瞬時靜止 (速率=0, 動能=0, 勢能=100)
    ep_real[apex_f] = 100.0
    ek_real[apex_f] = 0.0

    # 總機械能 Total = Ep + Ek
    total_energy = ep_real + ek_real
    for f in range(start_f, end_f + 1):
        total_energy[f] = min(100.0, max(total_energy[f], ep_real[f]))

    return {
        "start_f": start_f,
        "apex_f": apex_f,
        "end_f": end_f,
        "Ep": ep_real,
        "Ek": ek_real,
        "Total": total_energy,
        "speed_pct": np.sqrt(ek_real / 100.0) * 100.0,
        "real_h": real_h
    }


def get_f1_stage_info(current_frame, start_f, apex_f, end_f):
    """全面使用「速率 (Speed)」描述"""
    if current_frame < start_f:
        return "準備階段", "豆袋靜止在手中，未拋出", "手部高度 (最低基準)"
    elif current_frame == start_f:
        return "a. 剛拋出", "動能最大 (100% 最快 ↑)", "手部起點 (勢能 0%)"
    elif start_f < current_frame < apex_f:
        return "b. 向上飛", "速率變慢 ↘ (動能減少)", "高度升高 ↗ (勢能增加)"
    elif current_frame == apex_f:
        return "c. 最高點", "速率為 0 (瞬時靜止 ⊙)", "勢能最大 (100% 最高)"
    elif apex_f < current_frame < end_f:
        return "d. 向下墜", "速率加快 ↗ (動能增加)", "高度降低 ↘ (勢能減少)"
    elif current_frame == end_f:
        return "d. 接住豆袋", "動能最大 (100% 最快 ↓)", "落回手部 (勢能 0%)"
    else:
        return "運動結束", "豆袋已被雙手接住靜止", "手部高度 (靜止)"