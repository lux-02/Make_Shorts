import whisper
import sys
import os
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from moviepy.editor import (VideoFileClip, AudioFileClip, TextClip, ImageClip,
                            CompositeVideoClip, CompositeAudioClip, concatenate_videoclips,
                            ColorClip)
import moviepy.video.fx.all as vfx
from moviepy.config import change_settings

# ==========================================
# [설정 영역] 환경에 맞게 수정
# ==========================================
change_settings({"IMAGEMAGICK_BINARY": "/opt/homebrew/bin/magick"}) # 맥북 경로
FONT_PATH = "/Users/lux/Library/Fonts/SUIT-Bold.otf" 

# 파일 설정
DEFAULT_AUDIO_PATH = "download.wav"
VIDEO_FOLDER = "vd"  # 비디오 파일이 있는 폴더
BGM_PATH = "bg.mp3"  # 배경음악 파일 (선택사항)
BGM_VOLUME = 0.15    # 배경음악 볼륨 (0.0 ~ 1.0, 낮을수록 작음)

# 화면 및 자막 설정
TARGET_SIZE = (1080, 1920) # 숏츠 해상도
FONT_SIZE = 65
MAX_LINE_CHARS = 20 # 한 줄 최대 글자수(단어 단위 표시)
TRANSITION_DURATION = 0.0 # 하드 컷 (속도 맞춤이라 끊김 없이 연결됨)
MIN_SPEED = 0.8  # 너무 느리면 부자연스러움
MAX_SPEED = 1.3  # 너무 빠르면 부자연스러움
SUBTITLE_PAD = 0.0  # 자막 여유 시간 (겹침 방지)
# ==========================================

def fit_video_to_audio(video_path, target_duration):
    """
    영상을 오디오 길이(target_duration)에 강제로 맞추는 함수.
    길면 배속(Fast), 짧으면 슬로우(Slow)를 적용함 (CapCut 방식).
    """
    clip = VideoFileClip(video_path)
    
    # 1. 화면 꽉 차게 리사이즈 (Center Crop)
    # 현재 영상 비율(9:16)이 맞지만, 해상도가 낮으므로(416x752) 1080x1920으로 늘림
    ratio_w = TARGET_SIZE[0] / clip.w
    ratio_h = TARGET_SIZE[1] / clip.h
    scale_factor = max(ratio_w, ratio_h)
    
    clip = clip.resize(scale_factor)
    clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, 
                     width=TARGET_SIZE[0], height=TARGET_SIZE[1])
    
    # 2. 속도 조절 (Speed Ramping)
    # 목표 시간보다 영상이 길면 속도를 높이고(>1.0), 짧으면 속도를 낮춤(<1.0)
    original_duration = clip.duration
    target_duration = max(0.05, target_duration)
    speed_factor = original_duration / target_duration

    # 너무 큰 배속은 컷 편집으로 완화
    if speed_factor > MAX_SPEED:
        # 필요한 길이 계산 (MAX_SPEED 배속으로 재생했을 때 target_duration이 되는 길이)
        needed_len = target_duration * MAX_SPEED
        needed_len = min(needed_len, original_duration)
        
        # 영상의 앞부분 20% 제외하고 중간~끝 부분 사용 (안정적인 구간)
        safe_start = original_duration * 0.2
        available_len = original_duration - safe_start
        
        if available_len >= needed_len:
            # 중간 부분 사용
            start_at = safe_start + (available_len - needed_len) / 2
        else:
            # 길이가 부족하면 앞부분부터 사용
            start_at = max(0.0, (original_duration - needed_len) / 2)
        
        clip = clip.subclip(start_at, start_at + needed_len)
        original_duration = clip.duration
        speed_factor = original_duration / target_duration

    # 너무 느린 배속은 허용치까지만 낮추고 나머지는 프레임 동결로 처리
    if speed_factor < MIN_SPEED:
        speed_factor = MIN_SPEED

    print(f"   ⚙️ 속도 조정: {original_duration:.2f}초 -> {target_duration:.2f}초 (배속: {speed_factor:.2f}x)")

    # moviepy vfx.speedx 적용
    final_clip = clip.fx(vfx.speedx, speed_factor)

    # 부족한 길이는 마지막 프레임으로 채움 (루프 방지)
    if final_clip.duration < target_duration:
        pad = target_duration - final_clip.duration
        t_final = max(0.0, final_clip.duration - 0.03)
        t_orig = max(0.0, clip.duration - 0.03)
        try:
            frame = final_clip.get_frame(t_final)
        except Exception:
            try:
                frame = clip.get_frame(t_orig)
            except Exception:
                # 최후의 수단: 검정 화면으로 패딩
                frame = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
        freeze_clip = ImageClip(frame).set_duration(pad)
        final_clip = concatenate_videoclips([final_clip, freeze_clip])

    # 미세한 오차 제거를 위해 duration 강제 고정
    final_clip = final_clip.set_duration(target_duration)

    return final_clip

def create_subtitle(text, duration):
    """자막바 생성 (단일 라인)"""
    txt_clip = TextClip(text, fontsize=FONT_SIZE, color='white',
                        font=FONT_PATH, method='label', align='center', kerning=-1)
    
    # 배경 박스 (검정 반투명)
    bg_w = txt_clip.w + 60
    bg_h = txt_clip.h + 40
    bg_clip = ColorClip(size=(bg_w, bg_h), color=(0,0,0)).set_opacity(1)
    
    # 합성 및 위치 지정
    sub_final = CompositeVideoClip([bg_clip, txt_clip.set_pos('center')])
    sub_final = sub_final.set_duration(duration).set_pos(('center', 1300)) # 화면 하단
    return sub_final

def normalize_word(word: str) -> str:
    word = word.strip().lower()
    return re.sub(r"[^0-9a-z가-힣]+", "", word)

def split_display_words(text: str) -> List[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]

def extract_whisper_words(segments: List[Dict]) -> List[Dict]:
    words = []
    for seg in segments:
        if 'words' in seg and seg['words']:
            for w in seg['words']:
                word = normalize_word(w.get('word', ''))
                if not word:
                    continue
                words.append({
                    "word": word,
                    "start": w.get('start', seg['start']),
                    "end": w.get('end', seg['end'])
                })
    return words

def align_script_lines(script_lines: List[str], whisper_words: List[Dict]) -> List[Dict]:
    pointer = 0
    aligned = []
    
    for line_idx, line in enumerate(script_lines):
        display_words = split_display_words(line)
        normalized_words = [normalize_word(w) for w in display_words]
        word_times: List[Optional[Tuple[float, float]]] = [None] * len(display_words)
        line_start = None
        line_end = None
        matched_count = 0

        for i, nw in enumerate(normalized_words):
            if not nw:
                continue
            
            # 현재 위치에서 앞뒤 5개 단어 범위 내에서 fuzzy 매칭
            best_match_idx = None
            search_start = max(0, pointer - 2)
            search_end = min(len(whisper_words), pointer + 10)
            
            for j in range(search_start, search_end):
                if whisper_words[j]["word"] == nw:
                    best_match_idx = j
                    break
            
            if best_match_idx is not None:
                w = whisper_words[best_match_idx]
                word_times[i] = (w["start"], w["end"])
                if line_start is None:
                    line_start = w["start"]
                line_end = w["end"]
                pointer = best_match_idx + 1
                matched_count += 1
            else:
                # 매칭 실패 시 부분 문자열로 재시도
                for j in range(search_start, search_end):
                    ww = whisper_words[j]["word"]
                    # 3글자 이상이고 부분 일치하면 매칭
                    if len(nw) >= 3 and (nw in ww or ww in nw):
                        w = whisper_words[j]
                        word_times[i] = (w["start"], w["end"])
                        if line_start is None:
                            line_start = w["start"]
                        line_end = w["end"]
                        pointer = j + 1
                        matched_count += 1
                        break

        aligned.append({
            "text": line,
            "display_words": display_words,
            "word_times": word_times,
            "start": line_start,
            "end": line_end,
            "matched_words": matched_count
        })
    
    return aligned

def resolve_line_timings(aligned: List[Dict], segments: List[Dict], audio_duration: float) -> List[Dict]:
    # word 기반 타이밍이 있으면 절대 덮어쓰지 않음!
    # 없는 경우에만 보완
    for i, info in enumerate(aligned):
        if info["start"] is None or info["end"] is None:
            # word 타임스탬프가 완전히 없는 경우만 세그먼트/균등분할 사용
            if i < len(segments):
                if info["start"] is None:
                    info["start"] = segments[i]["start"]
                if info["end"] is None:
                    info["end"] = segments[i]["end"]
            else:
                per = audio_duration / max(1, len(aligned))
                if info["start"] is None:
                    info["start"] = i * per
                if info["end"] is None:
                    info["end"] = min(audio_duration, (i + 1) * per)

    # 시간 검증 및 최소한의 보정만 수행
    for i, info in enumerate(aligned):
        start = info["start"]
        end = info["end"]
        
        # 음수 duration 방지
        if end <= start:
            end = start + 0.5
        
        info["start"], info["end"] = start, end

    # 라인 간 간격 추가 (자연스러운 휴식)
    for i in range(len(aligned) - 1):
        current_end = aligned[i]["end"]
        next_start = aligned[i + 1]["start"]
        
        # 간격이 너무 작으면 최소 간격 확보
        if next_start - current_end < 0.2:
            gap = (next_start + current_end) / 2
            aligned[i]["end"] = gap - 0.1
            aligned[i + 1]["start"] = gap + 0.1

    # 마지막 라인은 오디오 끝까지
    if aligned:
        aligned[-1]["end"] = max(aligned[-1]["end"], audio_duration)
    
    return aligned

def chunk_words_with_times(info: Dict, max_chars: int) -> List[Dict]:
    words = info["display_words"]
    word_times = info["word_times"]
    line_start = info["start"]
    line_end = info["end"]
    duration = max(0.01, line_end - line_start)
    total_words = max(1, len(words))

    # 1. 매칭되지 않은 단어의 타이밍 보간
    interpolated_times = list(word_times)  # 복사
    
    # 연속된 미매칭 구간을 찾아서 보간
    i = 0
    while i < len(interpolated_times):
        if interpolated_times[i] is None:
            # 미매칭 구간 시작
            start_idx = i
            while i < len(interpolated_times) and interpolated_times[i] is None:
                i += 1
            end_idx = i
            
            # 앞뒤 시간 찾기
            prev_time = None
            next_time = None
            
            if start_idx > 0 and interpolated_times[start_idx - 1] is not None:
                prev_time = interpolated_times[start_idx - 1][1]  # 이전 단어 끝
            else:
                prev_time = line_start
            
            if end_idx < len(interpolated_times) and interpolated_times[end_idx] is not None:
                next_time = interpolated_times[end_idx][0]  # 다음 단어 시작
            else:
                # 다음 매칭 단어가 없으면 라인 끝까지
                remaining = end_idx - start_idx
                next_time = prev_time + remaining * (duration / total_words)
            
            # 균등 분배
            gap = next_time - prev_time
            num_words = end_idx - start_idx
            word_duration = gap / num_words
            
            for j in range(start_idx, end_idx):
                offset = j - start_idx
                start = prev_time + offset * word_duration
                end = start + word_duration
                interpolated_times[j] = (start, end)
        else:
            i += 1

    # 2. Chunk 생성 - TTS pause 기반으로 자연스럽게 분할
    chunks = []
    current = []
    current_len = 0
    
    for idx, w in enumerate(words):
        if not current:
            current = [(idx, w)]
            current_len = len(w)
        else:
            # 이전 단어와 현재 단어 사이의 pause 체크
            prev_idx = current[-1][0]
            pause = 0.0
            
            if interpolated_times[idx] and interpolated_times[prev_idx]:
                # 이전 단어 끝 시간과 현재 단어 시작 시간의 차이
                pause = interpolated_times[idx][0] - interpolated_times[prev_idx][1]
            
            # 자연스러운 pause(0.2초 이상) 또는 최대 길이 초과 시 chunk 분할
            should_split = False
            
            if pause > 0.2:  # TTS에서 긴 pause가 있는 지점
                should_split = True
            elif current_len + 1 + len(w) > max_chars:  # 길이 초과
                should_split = True
            
            if should_split:
                chunks.append(current)
                current = [(idx, w)]
                current_len = len(w)
            else:
                current.append((idx, w))
                current_len += 1 + len(w)
    
    if current:
        chunks.append(current)

    # 3. Chunk 타이밍 계산 (TTS pause 기반 자연스러운 분할)
    out = []
    
    for chunk_idx, group in enumerate(chunks):
        idxs = [i for i, _ in group]
        chunk_text = " ".join([w for _, w in group])
        
        # pause 정보 저장 (디버깅용)
        pause_before = 0.0
        if chunk_idx > 0 and len(chunks[chunk_idx - 1]) > 0:
            prev_last_idx = chunks[chunk_idx - 1][-1][0]
            curr_first_idx = idxs[0]
            if interpolated_times[curr_first_idx] and interpolated_times[prev_last_idx]:
                pause_before = interpolated_times[curr_first_idx][0] - interpolated_times[prev_last_idx][1]
        
        # 보간된 타이밍 사용
        times = [interpolated_times[i] for i in idxs]
        chunk_start = min(t[0] for t in times)
        chunk_end = max(t[1] for t in times)
        
        # 라인 범위 내로 제한
        chunk_start = max(line_start, chunk_start)
        chunk_end = min(line_end, chunk_end)
        
        # 최소 duration 보장 (0.3초)
        min_duration = 0.3
        if chunk_end - chunk_start < min_duration:
            # 중간 지점 기준으로 확장
            mid = (chunk_start + chunk_end) / 2
            chunk_start = max(line_start, mid - min_duration / 2)
            chunk_end = min(line_end, mid + min_duration / 2)
            
            # 여전히 짧으면 끝까지 확장
            if chunk_end - chunk_start < min_duration:
                chunk_end = min(line_end, chunk_start + min_duration)

        out.append({
            "text": chunk_text,
            "start": chunk_start,
            "end": chunk_end
        })
    
    # 4. Chunk 간 간격 완전 제거 (깜빡거림 제거)
    for i in range(len(out) - 1):
        current_chunk = out[i]
        next_chunk = out[i + 1]
        
        gap = next_chunk["start"] - current_chunk["end"]
        
        if gap > 0.01:
            # 간격이 있으면 현재 chunk를 다음 chunk 시작까지 연장
            out[i]["end"] = next_chunk["start"]
        elif gap < -0.01:
            # 겹치면 현재 chunk를 다음 chunk 시작 직전까지로 조정
            out[i]["end"] = next_chunk["start"]
    
    return out

# ==========================================
# 메인 실행
# ==========================================
def load_video_files(folder_path):
    """비디오 폴더에서 순차적으로 비디오 파일 로드"""
    if not os.path.exists(folder_path):
        print(f"\n📁 '{folder_path}' 폴더가 없습니다. 자동으로 생성합니다...")
        try:
            os.makedirs(folder_path)
            print(f"✅ '{folder_path}' 폴더가 생성되었습니다.")
            print(f"   이 폴더에 1.mp4, 2.mp4, 3.mp4... 형식으로 비디오 파일을 넣어주세요.")
        except Exception as e:
            print(f"⛔️ 폴더 생성 실패: {e}")
        return []
    
    # 비디오 확장자
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    
    # 폴더 내 모든 파일 가져오기
    all_files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in video_extensions:
                all_files.append((file, file_path))
    
    # 파일명 기준 정렬 (1.mp4, 2.mp4, ... 순서)
    try:
        all_files.sort(key=lambda x: int(os.path.splitext(x[0])[0]))
    except ValueError:
        # 숫자로 정렬 불가능하면 알파벳 순
        all_files.sort()
    
    return [path for name, path in all_files]

def input_multiline_script():
    """여러 줄 대본 입력받기"""
    print("\n📝 대본 입력")
    print("=" * 50)
    print("여러 줄의 대본을 입력하세요.")
    print("각 줄이 하나의 장면이 됩니다.")
    print("입력 완료 후 빈 줄에서 Enter를 누르세요.")
    print("=" * 50)
    
    lines = []
    line_num = 1
    
    while True:
        try:
            line = input(f"[{line_num}] ")
            if line.strip() == "":
                if lines:  # 이미 입력된 내용이 있으면 종료
                    break
                else:  # 첫 줄이 빈 줄이면 계속
                    continue
            lines.append(line.strip())
            line_num += 1
        except EOFError:
            break
    
    return lines

if __name__ == "__main__":
    print("🚀 쇼츠 영상 자동 생성 시작")
    print("=" * 50)
    
    # 1. 오디오 파일 확인
    print(f"\n🎵 오디오 파일 경로 (기본값: {DEFAULT_AUDIO_PATH})")
    audio_input = input(f"Enter를 누르면 기본값 사용, 또는 파일 경로 입력: ").strip()
    AUDIO_PATH = audio_input if audio_input else DEFAULT_AUDIO_PATH
    
    if not os.path.exists(AUDIO_PATH):
        print(f"\n⛔️ 오류: 오디오 파일 '{AUDIO_PATH}'을(를) 찾을 수 없습니다!")
        sys.exit(1)
    
    print(f"✅ 오디오 파일: {AUDIO_PATH}")
    
    # 1-1. 배경음악 설정
    if os.path.exists(BGM_PATH):
        print(f"\n🎶 배경음악 발견: {BGM_PATH}")
        bgm_choice = input(f"배경음악을 추가하시겠습니까? (y/n, 기본값: y): ").strip().lower()
        
        if bgm_choice == 'n':
            BGM_PATH = None
            print("   ℹ️ 배경음악 없이 진행합니다")
        else:
            volume_input = input(f"BGM 볼륨 설정 (0.0~1.0, 기본값: {BGM_VOLUME}): ").strip()
            if volume_input:
                try:
                    BGM_VOLUME = float(volume_input)
                    BGM_VOLUME = max(0.0, min(1.0, BGM_VOLUME))  # 0~1 범위로 제한
                    print(f"   ✅ BGM 볼륨: {BGM_VOLUME * 100:.0f}%")
                except ValueError:
                    print(f"   ⚠️ 잘못된 입력. 기본값({BGM_VOLUME})을 사용합니다")
            else:
                print(f"   ✅ BGM 볼륨: {BGM_VOLUME * 100:.0f}% (기본값)")
    else:
        print(f"\n🎶 배경음악 파일 없음 ({BGM_PATH})")
        print("   ℹ️ TTS 음성만 사용합니다")
        BGM_PATH = None
    
    # 2. 대본 입력
    USER_SCRIPT = input_multiline_script()
    
    if not USER_SCRIPT:
        print("\n⛔️ 오류: 대본이 입력되지 않았습니다!")
        sys.exit(1)
    
    print(f"\n✅ {len(USER_SCRIPT)}개 라인 입력 완료")
    for i, line in enumerate(USER_SCRIPT, 1):
        print(f"   [{i}] {line[:50]}{'...' if len(line) > 50 else ''}")
    
    # 3. 비디오 파일 로드
    print(f"\n📂 '{VIDEO_FOLDER}' 폴더에서 비디오 파일 검색 중...")
    VIDEO_FILES = load_video_files(VIDEO_FOLDER)
    
    if not VIDEO_FILES:
        print("\n⛔️ 오류: 비디오 파일을 찾을 수 없습니다!")
        print(f"   '{VIDEO_FOLDER}' 폴더에 1.mp4, 2.mp4, 3.mp4... 형식으로 비디오 파일을 넣어주세요.")
        sys.exit(1)
    
    print(f"\n✅ {len(VIDEO_FILES)}개 비디오 파일 발견:")
    for i, video_file in enumerate(VIDEO_FILES, 1):
        filename = os.path.basename(video_file)
        print(f"   [{i}] {filename}")
    
    # 4. 대본과 비디오 수 확인
    if len(USER_SCRIPT) != len(VIDEO_FILES):
        print(f"\n⚠️ 경고: 대본 수({len(USER_SCRIPT)})와 영상 수({len(VIDEO_FILES)})가 일치하지 않습니다!")
        if len(USER_SCRIPT) > len(VIDEO_FILES):
            print(f"   영상이 {len(USER_SCRIPT) - len(VIDEO_FILES)}개 부족합니다. 마지막 영상이 재사용됩니다.")
        else:
            print(f"   영상이 {len(VIDEO_FILES) - len(USER_SCRIPT)}개 초과입니다. 일부 영상은 사용되지 않습니다.")
    else:
        print(f"\n✅ 대본과 영상 수가 일치합니다!")
    
    # 5. 시작 확인
    print("\n" + "=" * 50)
    print("📋 설정 요약:")
    print(f"   🎵 오디오: {AUDIO_PATH}")
    if BGM_PATH and os.path.exists(BGM_PATH):
        print(f"   🎶 배경음악: {BGM_PATH} (볼륨: {BGM_VOLUME * 100:.0f}%)")
    else:
        print(f"   🎶 배경음악: 없음")
    print(f"   📝 대본: {len(USER_SCRIPT)}줄")
    print(f"   🎬 영상: {len(VIDEO_FILES)}개")
    print("=" * 50)
    
    response = input("\n영상 생성을 시작하시겠습니까? (y/n): ").strip().lower()
    if response != 'y':
        print("작업이 취소되었습니다.")
        sys.exit(0)
    
    print("\n🎬 영상 생성 시작...")

    # 1. Whisper로 오디오 분석 (시간 정보 획득)
    # User Script가 있으므로, Whisper는 '시간(Timestamp)' 추출용으로만 사용합니다.
    model = whisper.load_model("base")
    try:
        result = model.transcribe(AUDIO_PATH, language='ko', word_timestamps=True)
    except TypeError:
        result = model.transcribe(AUDIO_PATH, language='ko')
    segments = result['segments']
    whisper_words = extract_whisper_words(segments)
    
    final_clips = []
    original_audio = AudioFileClip(AUDIO_PATH)
    audio_duration = original_audio.duration
    
    print(f"\n📊 분석 결과:")
    print(f"   📋 대본 라인 수: {len(USER_SCRIPT)}")
    print(f"   🎙 Whisper 인식 문장 수: {len(segments)}")
    print(f"   🔤 Whisper 단어 수: {len(whisper_words)}")
    print(f"   🎬 영상 파일 수: {len(VIDEO_FILES)}")
    
    if len(whisper_words) == 0:
        print("⚠️ Whisper에서 단어 타임스탬프를 추출하지 못했습니다. 문장 단위 타이밍을 사용합니다.")
    else:
        print("\n🔍 Whisper 세그먼트 상세:")
        for i, seg in enumerate(segments):
            print(f"   [{i+1}] {seg['start']:.2f}~{seg['end']:.2f}s: {seg['text']}")
    
    loop_count = len(USER_SCRIPT)
    aligned = align_script_lines(USER_SCRIPT, whisper_words)
    aligned = resolve_line_timings(aligned, segments, audio_duration)
    
    # 디버그: 타이밍 확인
    print("\n⏰ 라인별 타이밍 정보 (word 매칭 후):")
    for i, info in enumerate(aligned):
        if info["start"] is not None and info["end"] is not None:
            duration = info["end"] - info["start"]
            start_str = f"{info['start']:.2f}"
            end_str = f"{info['end']:.2f}"
        else:
            duration = 0
            start_str = "N/A"
            end_str = "N/A"
        matched = info.get("matched_words", 0)
        total = len(info["word_times"])
        print(f"   [{i+1}] {duration:.2f}초 ({start_str} ~ {end_str}): {info['text'][:40]}... [매칭: {matched}/{total}]")
    
    # 전체 비디오 클립 생성 - 다음 라인 시작 직전까지 연장 (자막과 동일한 로직)
    print("\n🎬 영상 클립 타이밍 조정:")
    
    # 1단계: 각 라인의 실제 영상 duration 계산 (다음 라인 시작 직전까지)
    video_durations = []
    for i in range(loop_count):
        start_t = aligned[i]["start"]
        
        if i < loop_count - 1:
            # 다음 라인 시작 직전까지 (간격 없음)
            next_start = aligned[i + 1]["start"]
            end_t = next_start
        else:
            # 마지막 라인은 오디오 끝까지
            end_t = audio_duration
        
        duration = end_t - start_t
        video_durations.append({
            "index": i,
            "start": start_t,
            "end": end_t,
            "duration": duration,
            "original_end": aligned[i]["end"]
        })
        
        gap_info = ""
        if i < loop_count - 1:
            gap = aligned[i + 1]["start"] - aligned[i]["end"]
            if gap > 0.1:
                gap_info = f" (+{gap:.2f}초 연장)"
        
        print(f"   라인 [{i+1}]: {start_t:.2f}s ~ {end_t:.2f}s = {duration:.2f}초{gap_info}")
    
    # 2단계: 영상 클립 생성
    for i in range(loop_count):
        line_info = aligned[i]
        text_line = line_info["text"]
        vd = video_durations[i]
        start_t = vd["start"]
        end_t = vd["end"]
        duration = vd["duration"]

        # 영상 파일 1:1 매핑 (라인 1 -> 1.mp4, 라인 2 -> 2.mp4, ...)
        if i < len(VIDEO_FILES):
            video_path = VIDEO_FILES[i]
        else:
            # 영상이 부족한 경우 마지막 영상 재사용
            video_path = VIDEO_FILES[-1]
            print(f"   ⚠️ [{i+1}번 라인] 영상이 부족하여 {VIDEO_FILES[-1]}을(를) 재사용합니다.")
        
        print(f"\n[{i+1}/{loop_count}] Scene 생성 중...")
        print(f"   📝 대사: {text_line}")
        print(f"   ⏱ 영상 시간: {duration:.2f}초 ({start_t:.2f} ~ {end_t:.2f})")
        print(f"   🎙 TTS 시간: {vd['original_end'] - start_t:.2f}초 ({start_t:.2f} ~ {vd['original_end']:.2f})")
        print(f"   📼 영상: {video_path} (라인 {i+1} -> {video_path})")
        
        # 영상 가공 (CapCut Style: Time Stretch) - 연장된 duration 사용
        video_clip = fit_video_to_audio(video_path, duration)
        final_clips.append(video_clip)
    
    # 전체 영상 연결
    print("\n🎞  전체 영상 렌더링 준비 중...")
    base_video = concatenate_videoclips(final_clips, method="compose")
    
    # 전체 타임라인 기준으로 자막 생성
    print("\n📝 자막 생성 중...")
    
    # 1단계: 모든 chunk 정보 수집
    all_chunk_infos = []
    for i, line_info in enumerate(aligned):
        chunk_infos = chunk_words_with_times(line_info, MAX_LINE_CHARS)
        for chunk in chunk_infos:
            all_chunk_infos.append({
                "text": chunk["text"],
                "start": chunk["start"],
                "end": chunk["end"],
                "line_idx": i
            })
    
    # 2단계: 각 chunk의 end 시간을 다음 chunk 시작 직전까지 연장 (간격 0초)
    for i in range(len(all_chunk_infos)):
        chunk = all_chunk_infos[i]
        
        if i < len(all_chunk_infos) - 1:
            # 다음 chunk가 있으면 그 시작 직전까지 연장 (간격 없음)
            next_chunk = all_chunk_infos[i + 1]
            chunk["end"] = next_chunk["start"]
        else:
            # 마지막 chunk는 오디오 끝까지
            chunk["end"] = audio_duration
        
        # 최소 duration 보장 (0.4초로 완화)
        min_duration = 0.4
        if chunk["end"] - chunk["start"] < min_duration:
            # 다음 chunk 시작 직전까지 연장 (단, 최소 시간 보장)
            if i < len(all_chunk_infos) - 1:
                max_end = all_chunk_infos[i + 1]["start"]
                chunk["end"] = min(chunk["start"] + min_duration, max_end)
            else:
                chunk["end"] = min(chunk["start"] + min_duration, audio_duration)
    
    # 3단계: 자막 생성 및 출력
    all_subtitles = []
    prev_line_idx = -1
    
    for i, chunk in enumerate(all_chunk_infos):
        chunk_start = chunk["start"]
        chunk_end = chunk["end"]
        chunk_duration = chunk_end - chunk_start
        
        # 라인 구분 출력
        if chunk["line_idx"] != prev_line_idx:
            print(f"\n   라인 [{chunk['line_idx']+1}]: {aligned[chunk['line_idx']]['text'][:40]}...")
            prev_line_idx = chunk["line_idx"]
        
        # 이상 체크 및 pause 정보
        info_items = []
        if chunk_duration < 0.4:
            info_items.append(f"⚠️ SHORT")
        if i > 0:
            gap = chunk_start - all_chunk_infos[i-1]["end"]
            if gap > 0.01:
                info_items.append(f"GAP:{gap:.2f}s")
            elif gap < -0.01:
                info_items.append(f"⚠️ OVERLAP")
        
        info_str = f" [{', '.join(info_items)}]" if info_items else ""
        
        sub = create_subtitle(chunk["text"], chunk_duration).set_start(chunk_start)
        all_subtitles.append(sub)
        print(f"      '{chunk['text']}' ({chunk_start:.2f}s ~ {chunk_end:.2f}s = {chunk_duration:.2f}s){info_str}")
    
    print(f"\n총 {len(all_subtitles)}개 자막 생성됨")
    
    # 자막 합성
    final_video = CompositeVideoClip([base_video] + all_subtitles)
        
    # 오디오 합성 (TTS + BGM)
    print("\n🎵 오디오 합성 중...")
    
    # BGM 추가 여부 확인
    if BGM_PATH and os.path.exists(BGM_PATH):
        print(f"   ✅ 배경음악: {BGM_PATH}")
        print(f"   🔉 BGM 볼륨: {BGM_VOLUME * 100:.0f}%")
        
        try:
            # BGM 로드
            bgm = AudioFileClip(BGM_PATH)
            
            # BGM 길이를 영상 길이에 맞춤
            if bgm.duration < audio_duration:
                # BGM이 짧으면 루프
                num_loops = int(audio_duration / bgm.duration) + 1
                print(f"   🔁 BGM 루프: {num_loops}회 반복")
                bgm_clips = [bgm] * num_loops
                from moviepy.editor import concatenate_audioclips
                bgm = concatenate_audioclips(bgm_clips)
            
            # 정확한 길이로 자르기
            bgm = bgm.subclip(0, min(bgm.duration, audio_duration))
            
            # BGM 볼륨 조정
            bgm = bgm.volumex(BGM_VOLUME)
            
            # TTS와 BGM 믹싱
            final_audio = CompositeAudioClip([original_audio, bgm])
            final_video = final_video.set_audio(final_audio)
            print("   ✅ TTS + BGM 믹싱 완료")
        except Exception as e:
            print(f"   ⚠️ BGM 추가 실패: {e}")
            print("   ℹ️ TTS 음성만 사용합니다")
            final_video = final_video.set_audio(original_audio)
    else:
        # BGM 없으면 TTS만 사용
        print("   ℹ️ 배경음악 없음")
        print("   ℹ️ TTS 음성만 사용합니다")
        final_video = final_video.set_audio(original_audio)
    
    # 길이 정확히 맞추기
    if final_video.duration > audio_duration:
        final_video = final_video.subclip(0, audio_duration)
    elif final_video.duration < audio_duration:
        print(f"   ⚠️ 영상 길이({final_video.duration:.2f}s)가 오디오({audio_duration:.2f}s)보다 짧습니다.")
    
    # 내보내기
    print(f"\n💾 최종 영상 길이: {final_video.duration:.2f}초 (오디오: {audio_duration:.2f}초)")
    final_video.write_videofile("final_shorts_autofit.mp4", 
                                fps=30, 
                                codec="libx264", 
                                audio_codec="aac",
                                threads=4,
                                preset='medium')
    
    print("\n✨ 완성되었습니다! 'final_shorts_autofit.mp4' 확인")