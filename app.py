"""
app.py: 中一科學「實驗 5.2：研究向上拋豆袋的勢能與動能」
- 使用 session_state 快取影片分析數據，徹底解決重複逐幀分析的延遲問題
- 淺褐色高辨識度圓形播放/暫停按鈕
- 可直接拖動跳轉的影片播放進度時間軸 (Scrubber Slider)
- 紅色滑桿線精確橫切圓形按鈕正中心 (Vertical Center Alignment)
- 影片畫面與數據看板 1:1 等寬佈局
- 「較前」與「較後」單幀微調按鈕
- 100% 真實逐幀物理數據分析
"""

import tempfile
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from tracker import BeanbagTracker
from physics import (
    smooth_and_interpolate,
    calculate_real_frame_physics,
    get_f1_stage_info
)

st.set_page_config(
    page_title="實驗 5.2：研究向上拋豆袋的勢能與動能",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 全局基底字體 */
    html, body, [class*="css"] {
        font-size: 1.05rem !important;
    }

    h1 {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.2rem !important;
    }
    .sub-caption {
        font-size: 1.15rem !important;
        color: #94a3b8 !important;
        margin-bottom: 0.8rem !important;
    }

    /* 頂部 5 個微調按鈕標準樣式 */
    div[data-testid="stButton"] button {
        min-height: 44px !important;
        font-size: 1.08rem !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
    }

    /* 專屬：淺褐色圓形播放/暫停按鈕 */
    button[data-testid="baseButton-primary"],
    .st-key-btn_play_circle button,
    button[kind="primary"] {
        background: linear-gradient(135deg, #d49a5b 0%, #8c531b 100%) !important;
        background-color: #b8824f !important;
        color: #ffffff !important;
        border: 2.5px solid #fde68a !important;
        border-radius: 50% !important;
        width: 48px !important;
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        min-width: 48px !important;
        max-width: 48px !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 0 16px rgba(212, 154, 91, 0.75), 0 4px 10px rgba(0, 0, 0, 0.5) !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    button[data-testid="baseButton-primary"]:hover,
    .st-key-btn_play_circle button:hover,
    button[kind="primary"]:hover {
        transform: scale(1.10) !important;
        background: linear-gradient(135deg, #e4aa6b 0%, #9e5f22 100%) !important;
        background-color: #c48b54 !important;
        border-color: #ffffff !important;
        box-shadow: 0 0 22px rgba(228, 170, 107, 0.95), 0 6px 14px rgba(0, 0, 0, 0.6) !important;
    }

    button[data-testid="baseButton-primary"] p,
    .st-key-btn_play_circle button p,
    button[kind="primary"] p {
        color: #ffffff !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        line-height: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* 播放時間軸進度文字 */
    .play-progress-label {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 8px;
        margin-bottom: 2px;
    }

    /* 紅色滑桿軌道垂直居中微調 */
    div[data-testid="stSlider"] {
        padding-top: 19px !important;
        margin-top: 0 !important;
    }

    /* 影片顯示 */
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center !important;
        width: 100%;
    }
    div[data-testid="stImage"] img {
        max-height: 560px !important;
        width: 100% !important;
        object-fit: contain !important;
        border-radius: 14px;
        background-color: #0b1120;
    }

    /* 側邊數據看板：1:1 佈局 */
    .side-card-compact {
        background-color: #0f172a;
        border-radius: 12px;
        padding: 12px 14px;
        border: 1.5px solid #334155;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        display: flex;
        flex-direction: column;
        gap: 9px;
        width: 100%;
        box-sizing: border-box;
    }
    .side-item-compact {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 9px 12px;
        border-left: 4px solid #38bdf8;
        box-sizing: border-box;
        overflow: hidden;
    }
    
    /* 水平左右佈局 */
    .side-row-layout {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        width: 100%;
        box-sizing: border-box;
    }
    .side-text-col {
        flex: 0 0 46%;
        min-width: 110px;
    }
    .side-bar-col {
        flex: 1;
        min-width: 0;
        display: flex;
        flex-direction: column;
        gap: 3px;
        box-sizing: border-box;
    }
    .side-label {
        font-size: 0.90rem;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 1px;
    }
    .side-val-main {
        font-size: 1.35rem;
        font-weight: 800;
        line-height: 1.15;
    }
    .side-val-sub {
        font-size: 0.82rem;
        color: #cbd5e1;
        font-weight: 500;
        margin-top: 1px;
    }

    /* 橫向長條圖 */
    .meter-bar-track {
        width: 100%;
        height: 11px;
        background-color: #0b1120;
        border: 1px solid #334155;
        border-radius: 6px;
        overflow: hidden;
        position: relative;
        box-sizing: border-box;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.6);
    }
    .meter-bar-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.12s ease-out;
    }
    .meter-bar-scale {
        display: flex;
        justify-content: space-between;
        font-size: 0.66rem;
        color: #64748b;
        font-weight: 600;
        padding: 0 1px;
        line-height: 1;
        margin-top: 1px;
    }
    .side-badge {
        background-color: #0b1120;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 2px 8px;
        color: #94a3b8;
        font-size: 0.84rem;
        font-weight: 700;
        white-space: nowrap;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- 側邊欄：上傳影片 ----------------
st.sidebar.title("📁 實驗影片上傳")
uploaded_file = st.sidebar.file_uploader(
    "請上傳 iPad 拍攝的慢動作影片 (.MOV / .MP4)", 
    type=["mov", "mp4"]
)

# ---------------- 主標題 ----------------
st.title("實驗 5.2：研究向上拋豆袋的勢能與動能")
st.markdown('<div class="sub-caption">中一科學單元五：能量轉換與守恆 | 100% 逐幀真實數據分析</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.info("👈 請於左側欄上傳 iPad 拍攝的「向上拋豆袋」慢動作影片以開始分析。")
    st.stop()


# ---------------- 關鍵快取機制：分析僅執行一次，快取至 session_state ----------------
file_id = f"{uploaded_file.name}_{uploaded_file.size}"

if "cached_file_id" not in st.session_state or st.session_state.cached_file_id != file_id:
    with st.spinner("🤖 AI 正在逐幀提取物體真實像素座標，計算勢能與動能 (僅需分析一次)..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.getvalue())
            video_path = tfile.name

        tracker = BeanbagTracker()
        raw_frames, fps, trajectory = tracker.track_video(video_path, method="HSV")

        if len(raw_frames) == 0:
            st.error("無法解析影片，請確認影片格式為支援的 MP4/MOV。")
            st.stop()

        smoothed_y = smooth_and_interpolate(trajectory["y"], trajectory["valid"])
        physics_data = calculate_real_frame_physics(smoothed_y, trajectory["valid"], fps)

        # 寫入 Session State
        st.session_state.cached_frames = raw_frames
        st.session_state.cached_fps = fps
        st.session_state.cached_trajectory = trajectory
        st.session_state.cached_smoothed_y = smoothed_y
        st.session_state.cached_physics_data = physics_data
        st.session_state.cached_file_id = file_id
        st.session_state.video_frame = int(physics_data["apex_f"])
        st.session_state.is_playing = False

# 讀取快取數據（零延遲，不再重跑 OpenCV）
frames = st.session_state.cached_frames
fps = st.session_state.cached_fps
trajectory = st.session_state.cached_trajectory
smoothed_y = st.session_state.cached_smoothed_y
physics_data = st.session_state.cached_physics_data

total_frames = len(frames)
start_f = physics_data["start_f"]
apex_f = physics_data["apex_f"]
end_f = physics_data["end_f"]


# ---------------- 狀態管理與回調函數 ----------------
if "video_frame" not in st.session_state or st.session_state.video_frame >= total_frames:
    st.session_state.video_frame = int(apex_f)

if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

def jump_to_frame(target_frame):
    st.session_state.video_frame = int(target_frame)
    st.session_state.is_playing = False

def step_frame(step_amount):
    st.session_state.video_frame = max(0, min(total_frames - 1, st.session_state.video_frame + step_amount))
    st.session_state.is_playing = False

def toggle_playback():
    if st.session_state.is_playing:
        st.session_state.is_playing = False
    else:
        if st.session_state.video_frame >= total_frames - 1:
            st.session_state.video_frame = 0
        st.session_state.is_playing = True
        st.session_state.play_just_started = True

def on_slider_change():
    st.session_state.is_playing = False


# ---------------- 播放步進（在 Slider 實例化前執行） ----------------
if st.session_state.is_playing:
    if st.session_state.get("play_just_started", False):
        st.session_state.play_just_started = False
    else:
        if st.session_state.video_frame < total_frames - 1:
            st.session_state.video_frame += 1
        else:
            st.session_state.is_playing = False


# ---------------- 5 個微調按鈕 ----------------
b1, b2, b3, b4, b5 = st.columns(5)
with b1:
    st.button("⏮ 剛拋出點", on_click=jump_to_frame, args=(start_f,), use_container_width=True)
with b2:
    st.button("◀ 較前", on_click=step_frame, args=(-1,), use_container_width=True)
with b3:
    st.button("⊙ 最高點 (速率=0)", on_click=jump_to_frame, args=(apex_f,), use_container_width=True)
with b4:
    st.button("較後 ▶", on_click=step_frame, args=(1,), use_container_width=True)
with b5:
    st.button("⏭ 接住豆袋", on_click=jump_to_frame, args=(end_f,), use_container_width=True)


# ---------------- 播放按鈕與進度滑桿 ----------------
progress_ratio = float(st.session_state.video_frame) / float(max(1, total_frames - 1))
progress_ratio = min(1.0, max(0.0, progress_ratio))

st.markdown(
    f'<div class="play-progress-label">🎥 播放進度：第 <b>{st.session_state.video_frame}</b> / {total_frames - 1} 幀 ({int(progress_ratio * 100)}%) '
    f'<span style="color: #94a3b8; font-size: 0.95rem; font-weight: normal;">（可直接拖動滑桿跳轉影格）</span></div>',
    unsafe_allow_html=True
)

col_play, col_slider = st.columns([1, 16], gap="small")
with col_play:
    play_icon = "⏸" if st.session_state.is_playing else "▶"
    st.button(
        play_icon,
        on_click=toggle_playback,
        key="btn_play_circle",
        type="primary"
    )

with col_slider:
    selected_f = st.slider(
        "影片影格時間軸",
        min_value=0,
        max_value=total_frames - 1,
        key="video_frame",
        label_visibility="collapsed",
        on_change=on_slider_change
    )

# 當前幀數據
cur_h = physics_data["Ep"][selected_f]
cur_speed = physics_data["speed_pct"][selected_f]
cur_ep = physics_data["Ep"][selected_f]
cur_ek = physics_data["Ek"][selected_f]
cur_total = physics_data["Total"][selected_f]
stage_name, speed_desc, height_desc = get_f1_stage_info(selected_f, start_f, apex_f, end_f)
total_desc = "保持守恆 (100%)" if cur_total > 5 else "靜止於手中 (0%)"

h_bar_pct = max(0.0, min(100.0, float(cur_h)))
speed_bar_pct = max(0.0, min(100.0, float(cur_speed)))
total_bar_pct = max(0.0, min(100.0, float(cur_total)))


# ---------------- 1:1 佈局 ----------------
col_video, col_panel = st.columns([1, 1], gap="medium")

with col_video:
    curr_frame_rgb = frames[selected_f].copy()
    pil_img = Image.fromarray(curr_frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    cx = int(trajectory["x"][selected_f])
    cy = int(smoothed_y[selected_f])
    r = int(trajectory["radius"][selected_f]) if trajectory["radius"][selected_f] > 0 else 24

    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline="#00ffff", width=4)

    arrow_len = int((cur_speed / 100.0) * 80)
    if start_f <= selected_f < apex_f and arrow_len > 8:
        draw.line([(cx, cy), (cx, cy - arrow_len)], fill="#3b82f6", width=5)
        draw.polygon([(cx, cy - arrow_len - 8), (cx - 8, cy - arrow_len + 4), (cx + 8, cy - arrow_len + 4)], fill="#3b82f6")
    elif apex_f < selected_f <= end_f and arrow_len > 8:
        draw.line([(cx, cy), (cx, cy + arrow_len)], fill="#3b82f6", width=5)
        draw.polygon([(cx, cy + arrow_len + 8), (cx - 8, cy + arrow_len - 4), (cx + 8, cy + arrow_len - 4)], fill="#3b82f6")
    elif selected_f == apex_f:
        draw.ellipse([(cx - 10, cy - 10), (cx + 10, cy + 10)], fill="#eab308")

    st.image(pil_img, use_container_width=True)

with col_panel:
    st.markdown(f"""
    <div class="side-card-compact">
        <div class="side-item-compact" style="border-left-color: #f43f5e;">
            <div class="side-row-layout">
                <div class="side-text-col" style="flex: 1;">
                    <div class="side-label">📍 運動階段</div>
                    <div class="side-val-main" style="color: #f8fafc;">{stage_name}</div>
                </div>
                <div class="side-badge">第 {selected_f} 幀</div>
            </div>
        </div>
        <div class="side-item-compact" style="border-left-color: #ef4444;">
            <div class="side-row-layout">
                <div class="side-text-col">
                    <div class="side-label">📏 高度 (勢能 Ep)</div>
                    <div class="side-val-main" style="color: #ef4444;">{cur_h:.0f}%</div>
                    <div class="side-val-sub">{height_desc}</div>
                </div>
                <div class="side-bar-col">
                    <div class="meter-bar-track">
                        <div class="meter-bar-fill" style="width: {h_bar_pct:.1f}%; background: linear-gradient(90deg, #f87171, #ef4444);"></div>
                    </div>
                    <div class="meter-bar-scale">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="side-item-compact" style="border-left-color: #3b82f6;">
            <div class="side-row-layout">
                <div class="side-text-col">
                    <div class="side-label">⚡ 速率 (動能 Ek)</div>
                    <div class="side-val-main" style="color: #3b82f6;">{cur_speed:.0f}%</div>
                    <div class="side-val-sub">{speed_desc}</div>
                </div>
                <div class="side-bar-col">
                    <div class="meter-bar-track">
                        <div class="meter-bar-fill" style="width: {speed_bar_pct:.1f}%; background: linear-gradient(90deg, #60a5fa, #3b82f6);"></div>
                    </div>
                    <div class="meter-bar-scale">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="side-item-compact" style="border-left-color: #10b981;">
            <div class="side-row-layout">
                <div class="side-text-col">
                    <div class="side-label">🔄 總機械能 (Total)</div>
                    <div class="side-val-main" style="color: #e2e8f0;">{cur_total:.0f}%</div>
                    <div class="side-val-sub">{total_desc}</div>
                </div>
                <div class="side-bar-col">
                    <div class="meter-bar-track">
                        <div class="meter-bar-fill" style="width: {total_bar_pct:.1f}%; background: linear-gradient(90deg, #34d399, #10b981);"></div>
                    </div>
                    <div class="meter-bar-scale">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- 播放輪詢 ----------------
if st.session_state.is_playing:
    time.sleep(0.04)  # 25 fps
    st.rerun()