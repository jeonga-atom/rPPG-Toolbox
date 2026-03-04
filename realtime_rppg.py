import os
import cv2
import numpy as np
import pyrealsense2 as rs  # RealSense용 (웹캠이면 cv2.VideoCapture(0)으로 대체)
import matplotlib.pyplot as plt  # 필요 시 파형 플롯

# rPPG-Toolbox unsupervised POS 가져오기 (Toolbox 폴더 안에 있어야 함)
from unsupervised_methods.methods.POS import POS
from unsupervised_methods.utils import bandpass_filter, peak_detection

# POS 초기화 (fps는 RealSense 설정에 맞게)
pos_method = POS(fs=30.0)  # 30fps 가정

# RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 30fps RGB
pipeline.start(config)

# 얼굴 검출기 (OpenCV Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 버퍼 설정 (슬라이딩 윈도우: 180프레임 ≈ 6초)
buffer_size = 180
frame_buffer = []  # ROI 프레임 저장 (72x72x3)

print("실시간 rPPG 시작! 'q' 누르면 종료.")

try:
    while True:
        # RealSense 프레임 읽기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        frame = np.asanyarray(color_frame.get_data())  # BGR 이미지
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        bpm_text = "BPM: Calculating..."
        ppg_signal = None

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            # 얼굴 ROI 크롭 + Toolbox 스타일로 72x72 리사이즈
            roi = frame[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, (72, 72))
            roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)  # RGB로 변환 (POS 필요)

            # 버퍼에 추가
            frame_buffer.append(roi_rgb)
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)

            # 버퍼가 충분히 차면 PPG 계산
            if len(frame_buffer) == buffer_size:
                clip = np.stack(frame_buffer)  # (180, 72, 72, 3)

                # POS로 혈류 신호(PPG) 추출
                ppg_signal = pos_method(clip)  # 1D array

                # 밴드패스 필터 + 피크 검출로 BPM 계산
                filtered = bandpass_filter(ppg_signal, fs=30.0, low=0.75, high=3.0)
                peaks, _ = peak_detection(filtered)
                
                if len(peaks) > 5:  # 충분한 피크가 있어야 신뢰성 있음
                    rr_intervals = np.diff(peaks) / 30.0  # 초 단위
                    bpm = 60.0 / np.mean(rr_intervals)
                    bpm_text = f"BPM: {bpm:.1f}"
                else:
                    bpm_text = "BPM: Detecting..."

            # 얼굴 박스 그리기
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # BPM 텍스트 오버레이
        cv2.putText(frame, bpm_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # 화면 표시
        cv2.imshow('Real-time rPPG (POS method)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print("실시간 rPPG 종료.")