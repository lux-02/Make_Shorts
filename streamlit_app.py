import os
from typing import Optional

import streamlit as st

from main import (
    DEFAULT_AUDIO_PATH,
    VIDEO_FOLDER,
    BGM_FOLDER,
    BGM_LEGACY_PATH,
    BGM_VOLUME,
    BGM_EXTENSIONS,
    generate_shorts,
)


def pick_audio_file_dialog() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="오디오 파일 선택",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg"),
                ("All Files", "*.*"),
            ],
        )
        root.destroy()
        return selected or None
    except Exception:
        return None


def pick_directory_dialog(title: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=title)
        root.destroy()
        return selected or None
    except Exception:
        return None


st.set_page_config(page_title="쇼츠 자동 생성 GUI", layout="wide")
st.title("쇼츠 영상 자동 생성 GUI")
st.caption("기존 `main.py` 플로우를 GUI로 실행합니다.")

cwd = os.getcwd()
if "audio_path" not in st.session_state:
    st.session_state.audio_path = os.path.join(cwd, DEFAULT_AUDIO_PATH)
if "video_folder" not in st.session_state:
    st.session_state.video_folder = os.path.join(cwd, VIDEO_FOLDER)
if "bgm_folder" not in st.session_state:
    st.session_state.bgm_folder = os.path.join(cwd, BGM_FOLDER)
if "bgm_legacy_path" not in st.session_state:
    st.session_state.bgm_legacy_path = os.path.join(cwd, BGM_LEGACY_PATH)
if "manual_bgm_path" not in st.session_state:
    st.session_state.manual_bgm_path = ""
if "output_path" not in st.session_state:
    st.session_state.output_path = os.path.join(cwd, "final_shorts_autofit.mp4")
if "script_text" not in st.session_state:
    st.session_state.script_text = ""

st.subheader("1) 입력 경로")
col_a, col_b = st.columns([4, 1])
with col_a:
    st.text_input("오디오 파일 경로", key="audio_path")
with col_b:
    if st.button("오디오 선택", use_container_width=True):
        chosen = pick_audio_file_dialog()
        if chosen:
            st.session_state.audio_path = chosen
            st.rerun()
        st.info("선택을 취소했거나 선택기를 열 수 없습니다. 경로를 직접 입력할 수 있습니다.")

col_c, col_d = st.columns([4, 1])
with col_c:
    st.text_input("비디오 폴더 경로", key="video_folder")
with col_d:
    if st.button("폴더 선택", use_container_width=True):
        chosen = pick_directory_dialog("비디오 폴더 선택")
        if chosen:
            st.session_state.video_folder = chosen
            st.rerun()
        st.info("선택을 취소했거나 선택기를 열 수 없습니다. 경로를 직접 입력할 수 있습니다.")

st.subheader("2) 배경음악 설정")
use_bgm = st.checkbox("배경음악 사용", value=True)
bgm_volume = st.slider("BGM 볼륨", min_value=0.0, max_value=1.0, value=float(BGM_VOLUME), step=0.01)
st.text_input("BGM 폴더 경로 (랜덤 선택)", key="bgm_folder")
st.text_input("BGM 폴백 파일 경로", key="bgm_legacy_path")

col_e, col_f = st.columns([4, 1])
with col_e:
    st.text_input("직접 BGM 파일 지정 (선택)", key="manual_bgm_path")
with col_f:
    if st.button("BGM 선택", use_container_width=True):
        chosen = pick_audio_file_dialog()
        if chosen:
            st.session_state.manual_bgm_path = chosen
            st.rerun()

if use_bgm:
    manual = st.session_state.manual_bgm_path.strip()
    if manual:
        st.info(f"직접 지정 BGM 사용: `{manual}`")
    else:
        folder = st.session_state.bgm_folder.strip()
        legacy = st.session_state.bgm_legacy_path.strip()
        candidates = []
        if folder and os.path.isdir(folder):
            for name in os.listdir(folder):
                p = os.path.join(folder, name)
                if os.path.isfile(p) and name.lower().endswith(BGM_EXTENSIONS):
                    candidates.append(name)
        st.caption(f"BGM 후보 수: {len(candidates)}개")
        if candidates:
            st.caption("실행 시 후보 중 랜덤 1곡 선택")
        elif legacy and os.path.isfile(legacy):
            st.caption("폴더 후보 없음 -> 폴백 파일 사용")
        else:
            st.caption("사용 가능한 BGM 없음 -> TTS만 사용")

st.subheader("3) 대본")
st.text_area(
    "줄바꿈 기준으로 한 줄 = 한 장면",
    key="script_text",
    height=220,
    placeholder="첫 줄 대본\n둘째 줄 대본\n셋째 줄 대본",
)

st.subheader("4) 출력")
st.text_input("출력 파일 경로", key="output_path")

run = st.button("영상 생성 시작", type="primary", use_container_width=True)
log_placeholder = st.empty()

if run:
    script_lines = [line.strip() for line in st.session_state.script_text.splitlines() if line.strip()]
    audio_path = st.session_state.audio_path.strip()
    video_folder = st.session_state.video_folder.strip()
    output_path = st.session_state.output_path.strip()
    bgm_folder = st.session_state.bgm_folder.strip()
    bgm_legacy_path = st.session_state.bgm_legacy_path.strip()
    manual_bgm_path = st.session_state.manual_bgm_path.strip()

    if not audio_path:
        st.error("오디오 파일 경로를 입력하세요.")
        st.stop()
    if not video_folder:
        st.error("비디오 폴더 경로를 입력하세요.")
        st.stop()
    if not script_lines:
        st.error("대본을 1줄 이상 입력하세요.")
        st.stop()

    logs = []

    def ui_log(msg: str) -> None:
        logs.append(msg)
        log_placeholder.code("\n".join(logs[-250:]), language="text")

    selected_bgm_path: Optional[str] = None
    auto_pick_bgm = False
    if use_bgm:
        if manual_bgm_path:
            selected_bgm_path = manual_bgm_path
            auto_pick_bgm = False
        else:
            selected_bgm_path = None
            auto_pick_bgm = True

    try:
        result = generate_shorts(
            audio_path=audio_path,
            user_script=script_lines,
            video_folder=video_folder,
            selected_bgm_path=selected_bgm_path,
            bgm_volume=bgm_volume,
            output_path=output_path,
            bgm_folder=bgm_folder,
            bgm_legacy_path=bgm_legacy_path,
            auto_pick_bgm=auto_pick_bgm,
            log_fn=ui_log,
            moviepy_logger=None,
        )
        st.success(f"완료: `{result['output_path']}`")
        if os.path.exists(result["output_path"]):
            st.video(result["output_path"])
            with open(result["output_path"], "rb") as f:
                st.download_button(
                    "결과 파일 다운로드",
                    data=f,
                    file_name=os.path.basename(result["output_path"]),
                    mime="video/mp4",
                    use_container_width=True,
                )
    except Exception as e:
        st.error(f"실패: {e}")
