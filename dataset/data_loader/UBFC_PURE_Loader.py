"""
Hybrid loader for UBFC-rPPG and PURE-like directories.
BaseLoader를 상속해서 UBFC-rPPG and PURE 데이터셋의 디렉토리를 동시에 처리하도록 만든 커스텀 데이터 로더
main.py의 train/valid/test/unsupervised 로더 선택 분기에서 CONFIG.*.DATA.DATASET이 "UBFC-PURE"로 설정되면 이 클래스가 호출됨
"""
import csv
import glob
import json
import os
import re

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader


class UBFC_PURE_Loader(BaseLoader):
    """Flexible loader that handles both UBFC-rPPG and PURE folders."""

    def __init__(self, name, data_path, config_data, device=None):
        super().__init__(name, data_path, config_data, device)
        default_root = os.path.join(os.getcwd(), "data", "UBFC-rPPG")
        self.ubfc_original_root = getattr(
            self.config_data, "UBFC_ORIG_PATH", default_root)
        

    #*************************************************************************************************************
    # UBFC, PURE의 디렉터리 구조를 탐색하여 프레임 폴더와 라벨 파일이 있는 항목을 추려 바로 사용할 수 있는 dict 리스트로 정리하여 dataset_type을 붙여서 반환
    
    def get_raw_data(self, data_path):
        samples = []  # 최종 반환할 dict 리스트

        # <<<<UBFC>>>>
        # set으로 중복 제거, sorted로 오름차순 정렬 
        ubfc_paths = sorted(set(
            glob.glob(os.path.join(data_path, "subject*")) +
            glob.glob(os.path.join(data_path, "UBFC", "subject*"))
        ))
        for ubfc_dir in ubfc_paths:  # 경로 리스트만 순회
            frames_path = os.path.join(ubfc_dir, "frames")
            txt_candidates = sorted(glob.glob(os.path.join(ubfc_dir, "*.txt")))
            label_candidates = [
                p for p in txt_candidates
                if os.path.basename(p).lower().startswith(os.path.basename(ubfc_dir).lower())
            ]
            if os.path.isdir(frames_path) and label_candidates:
                samples.append({           # 결과 리스트에 append
                    "index": os.path.basename(ubfc_dir),
                    "path": ubfc_dir,
                    "dataset_type": "UBFC",
                    "label_path": label_candidates[0]
                })
        # <<<<PURE>>>>
        pure_paths = sorted(set(
            glob.glob(os.path.join(data_path, "*-*")) +
            glob.glob(os.path.join(data_path, "PURE", "*-*"))
        ))
        for pure_dir in pure_paths:        # 경로 리스트만 순회
            frames_path = os.path.join(pure_dir, "frames")
            label_candidates = glob.glob(os.path.join(pure_dir, "*.json"))
            if os.path.isdir(frames_path) and label_candidates:
                samples.append({           # 결과 리스트에 append
                    "index": os.path.basename(pure_dir),
                    "path": pure_dir,
                    "dataset_type": "PURE",
                    "label_path": label_candidates[0]
                })
        if not samples:
            raise ValueError(self.dataset_name + " data paths empty!")
        return samples
    

    #*************************************************************************************************************
    # get_raw_data로 얻은 dict 리스트에서 begin, end 비율에 맞게 샘플링하여 반환 (예: 0.0-0.8이면 처음 80% 샘플 반환)
    # 학습/ 검증용으로 나누는 함수 -> 데이터 분할
    # data_dirs는 list[dict] 형태

    def split_raw_data(self, data_dirs, begin, end):
        # 샘플을 퍼센트 기준으로 나눠 인덱스 변환
        if begin == 0 and end == 1:
            return data_dirs
        
        data_dirs_sorted = sorted(data_dirs, key=lambda item: item["index"])
        num_samples = len(data_dirs_sorted)
        start_idx = int(begin * num_samples)
        end_idx = int(end * num_samples)
        return data_dirs_sorted[start_idx:end_idx]


    #*************************************************************************************************************
    # i번째 샘플을 읽음 -> 프레임/라벨 전처리 -> 클립 단위로 저장 -> 저장된 클립 이름 딕셔너리 리스트 반환 -> file_list_dict[i]에 저장
    # 매 epoch마다 프레임을 읽고 필터링 하면 너무 느리기 때문에 학습을 빠르게 만들기 위한 오프라인 전처리 함수 -> 멀티프로세싱으로 병렬 처리 가능

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        data_dir = data_dirs[i]
        dataset_type = data_dir["dataset_type"]
        # filename = os.path.basename(data_dir["path"])                                         # 일단 안쓰기 때문에 주석처리
        saved_filename = data_dir["index"]                                                      # 저장할 때 쓸 파일 이름
        UBFC_ORIG_ROOT = "/home/jeonga/rPPG-Toolbox/data/UBFC-rPPG"                             # UBFC 원본 영상 경로

        if 'Motion' in config_preprocess.DATA_AUG:
            frames = self.read_npy_video(glob.glob(os.path.join(data_dir["path"], '*.npy')))    # npy 파일로 프레임 읽기
            sample_fs = getattr(self.config_data, 'FS', 30.0)                                   # 샘플 주파수
        else:
            frame_dir = os.path.join(data_dir["path"], "frames")                                # frame 파일에서 이미지들을 읽는 함수         
            frames = self.read_video(frame_dir)

            if dataset_type == "PURE":                                  # 데이터셋이 PURE이면 30Hz로 고정
                sample_fs = 30.0
            else:                                                       # UBFC이면 원본 fps 읽어서 사용 (읽기 실패하면 30Hz로 fallback)
                subj = data_dir["index"]  # "subject1"
                orig_vid = os.path.join(UBFC_ORIG_ROOT, subj, "vid.avi")# 원본 영상을 읽고
                sample_fs = self._read_video_fps(orig_vid)              # fps 읽기
                if not sample_fs or sample_fs <= 0:
                    sample_fs = 30.0                                    # 마지막 fallback (원하면 에러로 강제해도 됨)

        # 가짜 PPG 라벨을 생성하는 옵션 
        if config_preprocess.USE_PSUEDO_PPG_LABEL:                      # 라벨이 없거나 품질이 낮을 때, self-supervised 방식으로 학습/ 전처리
            bvps = self.generate_pos_psuedo_labels(frames, fs=sample_fs)
        # 실제 라벨 파일명이 *_waveform_30hz.txt 패턴. 
        # 없으면 바로 중단 => 라벨 없는 샘플은 학습 못함
        else:
            if dataset_type == "PURE":
                wave_candidates = glob.glob(os.path.join(data_dir["path"], "*_waveform_30hz.txt"))
                if not wave_candidates:
                    raise FileNotFoundError(
                        f"Can't find waveform file for {data_dir['path']}")
                wave_file = wave_candidates[0]
            else:
                wave_file = data_dir["label_path"]
                if not wave_file or not os.path.exists(wave_file):
                    raise FileNotFoundError(f"Label file not found for {data_dir['path']}")
            bvps = self.read_wave(wave_file)                            # self.read_wave 변환 타입: 보통 np.ndarray

        min_len = min(len(frames), len(bvps))

        if len(frames) != len(bvps):
            print(f"[Warning] Length mismatch: frames={len(frames)}, bvps={len(bvps)}. Cropping to {min_len}")

        frames = frames[:min_len]
        bvps = bvps[:min_len]

        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess, fs=sample_fs)           # 전처리하여 클립으로 자르기
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)# 클립들을 디스크에 저장 + 파일 리스트 만들기
        file_list_dict[i] = input_name_list                                                                 # file_list_dict[i](공유 딕셔너리)에 결과 기록        


#*******************************************<< 보조함수 >>***********************************************
# @staticmethod으로 선언된 함수들은 클래스(cls) 인스턴스(self)와 무관하게 독립적으로 사용할 수 있는 데코레이터
# 이런 static 메서드는 클래스명으로 바로 호출 할 수 있음

    # video보다는 frame 폴더에서 이미지들을 읽는 함수 
    # -> 프레임 리스트 반환
    @staticmethod
    def read_video(video_file):
        frame_paths = UBFC_PURE_Loader._list_frames(video_file)
        frames = []

        for frame_path in frame_paths:
            img = cv2.imread(frame_path)
            if img is None:
                continue
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not frames:
            raise FileNotFoundError(f"Unable to read any frames from {video_file}")
        return np.asarray(frames)


    # 프레임의 FPS읽는 함수 
    # -> UBFC 원본 영상에서 fps 읽어서 반환 (실패하면 0.0 반환)
    @staticmethod
    def _list_frames(frame_directory):
        if not os.path.isdir(frame_directory):
            raise FileNotFoundError(f"Frame directory missing: {frame_directory}")
        patterns = ['*.png', '*.jpg', '*.jpeg']
        frame_paths = []
        for pattern in patterns:
            frame_paths.extend(glob.glob(os.path.join(frame_directory, pattern)))
        if not frame_paths:
            raise FileNotFoundError(f"No supported frames found in {frame_directory}")
        def sort_key(path):
            basename = os.path.basename(path)
            nums = re.findall(r"\d+", basename)
            if nums:
                return int(nums[-1])
            return basename
        frame_paths = sorted(frame_paths, key=sort_key)
        return frame_paths


    # video의 fps를 읽는 함수
    @staticmethod
    def _read_video_fps(video_file):
        if not os.path.exists(video_file):
            return 0.0
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            duration_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if duration_ms > 0 and total_frames > 0:
                fps = total_frames / (duration_ms / 1000.0)
        cap.release()
        return float(fps) if fps and fps > 0 else 0.0

    def _resolve_ubfc_fps(self, subject):
        video_file = os.path.join(self.ubfc_original_root, subject, "vid.avi")
        if not os.path.exists(video_file):
            return 0.0
        return self._read_video_fps(video_file)


    # JSON/TXT/CSV 파일 형식을 읽어서 PPG 라벨 시퀀스 반환하는 함수
    @staticmethod
    def read_wave(bvp_file):
        if not os.path.exists(bvp_file):
            raise FileNotFoundError(f"Label file not found: {bvp_file}")
        _, ext = os.path.splitext(bvp_file)
        ext = ext.lower()
        if ext == '.json':
            with open(bvp_file, 'r') as f:
                labels = json.load(f)
                waves = [label["Value"]["waveform"] for label in labels["/FullPackage"]]
            return np.asarray(waves)
        if ext in ['.txt', '.csv']:
            values = []
            with open(bvp_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    for entry in row:
                        for token in entry.strip().split():
                            if not token:
                                continue
                            try:
                                values.append(float(token))
                            except ValueError:
                                continue
            if not values:
                raise ValueError(f"No numeric values found in {bvp_file}")
            return np.asarray(values)
        raise ValueError(f"Unsupported waveform extension {ext} in {bvp_file}")
