"""
app.py: 中一科學「實驗 5.2：研究向上拋豆袋的勢能與動能」
- 專為 iPad / iPadOS 17+ 觸控最佳化
- 解決 iPad 收起側邊欄後無法重新展開的問題（浮動專用展開按鈕）
- 徹底防止頂部按鈕文字截斷（響應式字體與自動換行支援）
- 100% 像素級還原純白雙垂直長方形暫停圖示（根除 iPad = Emoji 錯誤）
- 播放按鈕與紅色進度條增加安全邊距，徹底杜絕貼合問題
"""

import os
import io
import base64
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

# 輔助函式：安全注入純 HTML/CSS
def render_html(html_str: str):
    clean_lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    st.markdown("\n".join(clean_lines), unsafe_allow_html=True)


# ---------------- 全局 CSS 樣式 ----------------
render_html("""
<style>
/* 全局基底字體 */
html, body, [class*="css"] {
    font-size: 1.05rem !important;
}

h1 {
    font-size: 2.0rem !important;
    font-weight: 800 !important;
    margin-bottom: 0.2rem !important;
}
.sub-caption {
    font-size: 1.10rem !important;
    color: #94a3b8 !important;
    margin-bottom: 0.8rem !important;
}

/* 隱藏頂部彩虹裝飾條與多餘部署按鈕 */
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}
#MainMenu {
    display: none !important;
    visibility: hidden !important;
}
header [data-testid="stHeaderActionElements"],
header [data-testid="stToolbar"],
.stForkButton,
.stAppDeployButton {
    display: none !important;
    visibility: hidden !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}
footer {
    display: none !important;
    visibility: hidden !important;
}

/* ---------------- 側邊欄展開按鈕 (iPad 友善專屬修復) ---------------- */
/* 當點擊 << 收起側邊欄後，左上角常駐醒目的高對比度浮動按鈕，隨時可一鍵重新展開 */
div[data-testid="collapsedControl"],
div[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 10px !important;
    left: 10px !important;
    z-index: 9999999 !important;
    background-color: #0f172a !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 10px !important;
    padding: 6px 12px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6) !important;
    cursor: pointer !important;
    align-items: center !important;
    justify-content: center !important;
}

div[data-testid="collapsedControl"]:hover,
div[data-testid="stSidebarCollapsedControl"]:hover {
    background-color: #1e293b !important;
    border-color: #7dd3fc !important;
    transform: scale(1.04) !important;
}

div[data-testid="collapsedControl"] svg,
div[data-testid="stSidebarCollapsedControl"] svg {
    fill: #38bdf8 !important;
    stroke: #38bdf8 !important;
    color: #38bdf8 !important;
    width: 20px !important;
    height: 20px !important;
}

/* 展開按鈕附帶清晰繁體文字標籤 */
div[data-testid="collapsedControl"]::after,
div[data-testid="stSidebarCollapsedControl"]::after {
    content: " 📁 上傳影片" !important;
    color: #38bdf8 !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    margin-left: 6px !important;
    white-space: nowrap !important;
}

/* 側邊欄內的收起按鈕樣式 */
section[data-testid="stSidebar"] button[data-testid="stSidebarCollapseButton"] {
    background: #1e293b !important;
    border: 1.5px solid #475569 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}

/* ---------------- 頂部 5 個微調按鈕 (防止文字變 ... 省略號) ---------------- */
div[data-testid="stButton"] button {
    min-height: 42px !important;
    height: auto !important;
    padding: 5px 3px !important;
    border-radius: 10px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    line-height: 1.25 !important;
}

div[data-testid="stButton"] button p {
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    text-align: center !important;
    white-space: normal !important; /* 允許自動換行，徹底消除 ... */
    word-break: break-word !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* iPad / 平板螢幕尺寸微調 */
@media screen and (max-width: 1024px) {
    div[data-testid="stButton"] button {
        min-height: 40px !important;
        padding: 4px 2px !important;
    }
    div[data-testid="stButton"] button p {
        font-size: 0.82rem !important;
        letter-spacing: -0.3px !important;
    }
}

/* ---------------- 播放 / 暫停按鈕 (1:1 像素級還原截圖樣式) ---------------- */
button[data-testid="baseButton-primary"],
.st-key-btn_play_circle button,
.st-key-btn_stop_circle button,
button[kind="primary"] {
    background: linear-gradient(135deg, #d49a5b 0%, #8c531b 100%) !important;
    background-color: #b8824f !important;
    color: #ffffff !important;
    border: 2.5px solid #fde68a !important;
    border-radius: 14px !important; /* 還原截圖中的微圓角方形 (Squircle) */
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
    flex-shrink: 0 !important;
}

button[data-testid="baseButton-primary"]:hover,
.st-key-btn_play_circle button:hover,
.st-key-btn_stop_circle button:hover,
button[kind="primary"]:hover {
    transform: scale(1.08) !important;
    background: linear-gradient(135deg, #e4aa6b 0%, #9e5f22 100%) !important;
    border-color: #ffffff !important;
    box-shadow: 0 0 22px rgba(228, 170, 107, 0.95), 0 6px 14px rgba(0, 0, 0, 0.6) !important;
}

/* 播放箭頭樣式 */
.st-key-btn_play_circle button p {
    color: #ffffff !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
    padding-left: 3px !important; /* 光學中心微調 */
}

/* 暫停圖示：徹底屏蔽 iPad 原生 = Emoji，純 CSS 繪製兩條純白垂直長條 (100% 還原 MacBook 截圖) */
.st-key-btn_stop_circle button p {
    display: none !important;
    visibility: hidden !important;
}
.st-key-btn_stop_circle button::after {
    content: "" !important;
    display: block !important;
    width: 14px !important;
    height: 18px !important;
    border-left: 5px solid #ffffff !important;
    border-right: 5px solid #ffffff !important;
    box-sizing: border-box !important;
}

/* ---------------- 播放時間軸與滑桿 ---------------- */
.play-progress-label {
    font-size: 1.10rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 8px;
    margin-bottom: 2px;
}

/* 控制列垂直居中 */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stSlider"]) {
    align-items: center !important;
}

/* 紅色滑桿向上微調並垂直居中，強制與左側按鈕保持安全距離（徹底解決貼黏問題） */
div[data-testid="stSlider"] {
    padding-top: 0px !important;
    margin-top: -6px !important;
    margin-left: 20px !important; /* 關鍵：強制在左側隔離出 20px 空隙 */
    padding-left: 2px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

/* 加粗紅條軌道厚度至 8px，並設圓角 */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {
    height: 8px !important;
    border-radius: 4px !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    height: 8px !important;
    border-radius: 4px !important;
}

/* 拖曳圓點增大至 20px */
div[data-testid="stSlider"] [role="slider"] {
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.8) !important;
}

/* ---------------- 側邊數據看板 ---------------- */
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
    transition: width 0.10s ease-out;
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
""")


# ---------------- 側邊欄：影片上傳 ----------------
st.sidebar.title("📁 實驗影片上傳")
uploaded_file = st.sidebar.file_uploader(
    "請上傳 iPad 拍攝的慢動作影片 (.MOV / .MP4)", 
    type=["mov", "mp4"]
)

# ---------------- 主標題 ----------------
st.title("實驗 5.2：研究向上拋豆袋的勢能與動能")
render_html('<div class="sub-caption">中一科學單元五：能量轉換與守恆 | 100% 逐幀真實數據分析</div>')

if uploaded_file is None:
    for k in ["cached_frames", "cached_fps", "cached_trajectory", "cached_smoothed_y", "cached_physics_data", "cached_file_id"]:
        if k in st.session_state:
            del st.session_state[k]
    st.info("👈 請於左側欄上傳 iPad 拍攝的「向上拋豆袋」慢動作影片以開始分析。")
    st.stop()

file_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
video_bytes = uploaded_file.getvalue()


# ---------------- 分析影片 (僅在更換新影片時執行一次) ----------------
if "cached_file_id" not in st.session_state or st.session_state.cached_file_id != file_id:
    for k in ["cached_frames", "cached_fps", "cached_trajectory", "cached_smoothed_y", "cached_physics_data"]:
        if k in st.session_state:
            del st.session_state[k]
    import gc
    gc.collect()

    with st.spinner("🤖 AI 正在逐幀提取物體真實像素座標，計算勢能與動能 (分析約需 3~5 秒)..."):
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in [".mp4", ".mov"]:
            ext = ".mp4"

        video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tfile:
                tfile.write(video_bytes)
                video_path = tfile.name

            tracker = BeanbagTracker()
            raw_frames, fps, trajectory = tracker.track_video(video_path, method="HSV")
        finally:
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception:
                    pass

        if len(raw_frames) == 0:
            st.error("無法解析影片，請確認影片格式為支援的 MP4/MOV。")
            st.stop()

        smoothed_y = smooth_and_interpolate(trajectory["y"], trajectory["valid"])
        physics_data = calculate_real_frame_physics(smoothed_y, trajectory["valid"], fps)

        st.session_state.cached_frames = raw_frames
        st.session_state.cached_fps = fps
        st.session_state.cached_trajectory = trajectory
        st.session_state.cached_smoothed_y = smoothed_y
        st.session_state.cached_physics_data = physics_data
        st.session_state.cached_file_id = file_id
        st.session_state.video_frame = int(physics_data["apex_f"])
        st.session_state.is_playing = False

# 讀取快取數據
frames = st.session_state.cached_frames
fps = st.session_state.cached_fps
trajectory = st.session_state.cached_trajectory
smoothed_y = st.session_state.cached_smoothed_y
physics_data = st.session_state.cached_physics_data

total_frames = len(frames)
start_f = physics_data["start_f"]
apex_f = physics_data["apex_f"]
end_f = physics_data["end_f"]


# ---------------- 播放狀態與控制函式 ----------------
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

def start_playback():
    if st.session_state.video_frame >= total_frames - 1:
        st.session_state.video_frame = 0
    st.session_state.is_playing = True

def stop_playback():
    st.session_state.is_playing = False

def on_slider_change():
    st.session_state.is_playing = False


# ---------------- 播放推進步進 (合法位置：必須在 st.slider 呼叫前執行) ----------------
if st.session_state.is_playing:
    if st.session_state.video_frame < total_frames - 1:
        st.session_state.video_frame += 1
    else:
        st.session_state.video_frame = total_frames - 1
        st.session_state.is_playing = False


# ---------------- 5 個微調按鈕 (合理分配寬度比例，保證完全不截斷) ----------------
b1, b2, b3, b4, b5 = st.columns([1.15, 0.85, 1.45, 0.85, 1.15])
with b1:
    st.button("⏮ 剛拋出", on_click=jump_to_frame, args=(start_f,), use_container_width=True)
with b2:
    st.button("◀ 較前", on_click=step_frame, args=(-1,), use_container_width=True)
with b3:
    st.button("⊙ 最高點 (速率=0)", on_click=jump_to_frame, args=(apex_f,), use_container_width=True)
with b4:
    st.button("較後 ▶", on_click=step_frame, args=(1,), use_container_width=True)
with b5:
    st.button("⏭ 接住豆袋", on_click=jump_to_frame, args=(end_f,), use_container_width=True)


# ---------------- 播放時間軸與控制按鈕 ----------------
progress_ratio = float(st.session_state.video_frame) / float(max(1, total_frames - 1))
progress_ratio = min(1.0, max(0.0, progress_ratio))

render_html(
    f'<div class="play-progress-label">🎥 播放進度：第 <b>{st.session_state.video_frame}</b> / {total_frames - 1} 幀 ({int(progress_ratio * 100)}%) '
    f'<span style="color: #94a3b8; font-size: 0.95rem; font-weight: normal;">（可直接拖動滑桿跳轉影格）</span></div>'
)

# 調整欄位比例至 [1.1, 15]，配合 CSS margin-left 徹底隔開
col_play, col_slider = st.columns([1.1, 15], gap="small")
with col_play:
    if st.session_state.is_playing:
        st.button("⏸", on_click=stop_playback, key="btn_stop_circle", type="primary")
    else:
        st.button("▶", on_click=start_playback, key="btn_play_circle", type="primary")

with col_slider:
    selected_f = st.slider(
        "影片影格時間軸",
        min_value=0,
        max_value=total_frames - 1,
        key="video_frame",
        on_change=on_slider_change,
        label_visibility="collapsed"
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

    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG", quality=80)
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    render_html(
        f'''
        <div style="display: flex; justify-content: center; width: 100%;">
            <img src="data:image/jpeg;base64,{img_b64}" 
                 style="max-height: 560px; width: 100%; object-fit: contain; border-radius: 14px; background-color: #0b1120;" />
        </div>
        '''
    )

with col_panel:
    render_html(f"""
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
    """)


# ---------------- 播放輪詢與定時 ----------------
if st.session_state.is_playing:
    time.sleep(0.08)
    st.rerun()