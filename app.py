"""
app.py: 中一科學「實驗 5.2：研究向上拋豆袋的勢能與動能」
- 使用純淨 st.markdown 注入 HTML/CSS，杜絕動態模組載入錯誤 (Html.js 報錯)
- 遵循 Streamlit 官方 Session State 規範，在 slider 呼叫前合法更新步進，徹底解決 StreamlitWidgetAlreadyInstantiatedError
- 紅條軌道垂直加粗至 8px，向上提昇並與播放按鈕精確垂直居中對齊
- 拖曳時間軸紅條時，上方進度文字（幀數與百分比）0 延遲即時聯動更新 (Live-update)
- 碰觸時間軸瞬間停止背景播放，消除搶幀衝突
- 1:1 佈局、精準垂直居中、真實逐幀數據分析
"""

import os
import io
import base64
import tempfile
import time
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
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
    """
    1. 逐行去除前後空白 (line.strip())，徹底消除 4 格縮排，100% 杜絕 Markdown 誤判為程式碼區塊 (<pre><code>)。
    2. 過濾純空行，防止 CommonMark 在空行處提早終止 HTML 區塊解析。
    3. 絕不調用 st.html()，避免 Streamlit Cloud 動態模組載入失敗 (Failed to fetch dynamically imported module)。
    """
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
.st-key-btn_stop_circle button,
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
.st-key-btn_stop_circle button:hover,
button[kind="primary"]:hover {
    transform: scale(1.10) !important;
    background: linear-gradient(135deg, #e4aa6b 0%, #9e5f22 100%) !important;
    background-color: #c48b54 !important;
    border-color: #ffffff !important;
    box-shadow: 0 0 22px rgba(228, 170, 107, 0.95), 0 6px 14px rgba(0, 0, 0, 0.6) !important;
}

button[data-testid="baseButton-primary"] p,
.st-key-btn_play_circle button p,
.st-key-btn_stop_circle button p,
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

/* 播放控制列整行垂直完美居中 */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stSlider"]) {
    align-items: center !important;
}

/* 紅色滑桿向上提昇並精確垂直居中 */
div[data-testid="stSlider"] {
    padding-top: 0px !important;
    margin-top: -6px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
}

/* 垂直加粗紅條厚度至 8px，並設圓角 */
div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {
    height: 8px !important;
    border-radius: 4px !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    height: 8px !important;
    border-radius: 4px !important;
}

/* 拖曳圓點增大至 20px，發光手感提升 */
div[data-testid="stSlider"] [role="slider"] {
    width: 20px !important;
    height: 20px !important;
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.8) !important;
}

/* 非阻塞隱藏輔助 iframe，防止版面留白 */
div[data-testid="stCustomComponentV1"] {
    height: 0px !important;
    min-height: 0px !important;
    max-height: 0px !important;
    margin: 0px !important;
    padding: 0px !important;
    overflow: hidden !important;
    border: none !important;
}
iframe[data-testid="stCustomComponentV1"] {
    height: 0px !important;
    border: none !important;
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

# ---------------- 前端事件捕獲與即時聯動監聽 ----------------
components.html(
    """
    <script>
    (function() {
        try {
            const parentDoc = window.parent.document;
            if (!parentDoc) return;

            // 1. 碰觸滑桿瞬間暫停播放，防止搶幀
            if (!window.parent._stSliderTouchRegistered) {
                window.parent._stSliderTouchRegistered = true;
                const onSliderTouch = function(e) {
                    const slider = e.target.closest('[data-testid="stSlider"]');
                    if (!slider) return;

                    const stopBtn = parentDoc.querySelector('.st-key-btn_stop_circle button') ||
                                    parentDoc.querySelector('button[kind="primary"]');
                    if (stopBtn && (stopBtn.innerText.includes('⏸') || stopBtn.textContent.includes('⏸'))) {
                        stopBtn.click();
                    }
                };

                parentDoc.addEventListener('pointerdown', onSliderTouch, true);
                parentDoc.addEventListener('touchstart', onSliderTouch, true);
                parentDoc.addEventListener('mousedown', onSliderTouch, true);
            }

            // 2. 拖曳紅條時，進度文字 0 延遲即時更新 (Live-update)
            function setupLiveSync() {
                const slider = parentDoc.querySelector('[data-testid="stSlider"]');
                if (!slider) return;

                const thumb = slider.querySelector('[role="slider"]');
                if (!thumb) return;

                const lblFrame = parentDoc.getElementById('lbl-cur-frame');
                const lblPct = parentDoc.getElementById('lbl-cur-pct');

                function update() {
                    const now = thumb.getAttribute('aria-valuenow');
                    if (now === null || now === undefined) return;
                    const cur = parseInt(now, 10);
                    if (isNaN(cur)) return;
                    const maxVal = parseInt(thumb.getAttribute('aria-valuemax') || '151', 10);
                    const pct = maxVal > 0 ? Math.min(100, Math.max(0, Math.round((cur / maxVal) * 100))) : 0;
                    
                    if (lblFrame && lblFrame.textContent !== String(cur)) {
                        lblFrame.textContent = cur;
                    }
                    if (lblPct && lblPct.textContent !== (pct + '%')) {
                        lblPct.textContent = pct + '%';
                    }
                }

                if (!thumb._liveObs) {
                    thumb._liveObs = new MutationObserver(update);
                    thumb._liveObs.observe(thumb, { attributes: true, attributeFilter: ['aria-valuenow'] });
                }

                if (!slider._liveEvents) {
                    slider._liveEvents = true;
                    slider.addEventListener('pointermove', update, { passive: true });
                    slider.addEventListener('touchmove', update, { passive: true });
                    slider.addEventListener('mousemove', update, { passive: true });
                }
            }

            setupLiveSync();
            if (!window.parent._stSliderInterval) {
                window.parent._stSliderInterval = setInterval(setupLiveSync, 200);
            }
        } catch (err) {
            console.error("Slider listener init error:", err);
        }
    })();
    </script>
    """,
    height=0,
)


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
    with st.spinner("🤖 AI 正在逐幀提取物體真實像素座標，計算勢能與動能 (分析約需 4~6 秒，分析後即快取)..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(video_bytes)
            video_path = tfile.name

        tracker = BeanbagTracker()
        raw_frames, fps, trajectory = tracker.track_video(video_path, method="HSV")

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


# ---------------- 播放時間軸與控制按鈕 ----------------
progress_ratio = float(st.session_state.video_frame) / float(max(1, total_frames - 1))
progress_ratio = min(1.0, max(0.0, progress_ratio))

render_html(
    f'<div class="play-progress-label">🎥 播放進度：第 <b id="lbl-cur-frame">{st.session_state.video_frame}</b> / <span id="lbl-total-frames">{total_frames - 1}</span> 幀 (<span id="lbl-cur-pct">{int(progress_ratio * 100)}%</span>) '
    f'<span style="color: #94a3b8; font-size: 0.95rem; font-weight: normal;">（可直接拖動滑桿跳轉影格）</span></div>'
)

col_play, col_slider = st.columns([1, 16], gap="small")
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

    # 將圖像轉為 Base64 內嵌，消除非同步延遲
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