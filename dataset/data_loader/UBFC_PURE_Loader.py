"""
Hybrid loader for UBFC-rPPG, PURE-like, and VIPL ROI frame directories.
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
    """Flexible loader that handles UBFC-rPPG, PURE, and pre-cropped VIPL folders."""

    def __init__(self, name, data_path, config_data, device=None):
        default_root = os.path.join(os.getcwd(), "data", "UBFC-rPPG")
        self.ubfc_original_root = getattr(config_data, "UBFC_ORIG_PATH", default_root)
        super().__init__(name, data_path, config_data, device)
        

    #*************************************************************************************************************
    # UBFC, PURE, VIPL의 디렉터리 구조를 탐색하여 프레임 폴더와 라벨 파일이 있는 항목을 추려 반환
    
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
            label_candidates = (
                sorted(glob.glob(os.path.join(pure_dir, "*_waveform_30hz.txt"))) +
                sorted(glob.glob(os.path.join(pure_dir, "*.json")))
            )
            if os.path.isdir(frames_path) and label_candidates:
                samples.append({           # 결과 리스트에 append
                    "index": os.path.basename(pure_dir),
                    "path": pure_dir,
                    "dataset_type": "PURE",
                    "label_path": label_candidates[0]
                })

        # <<<<VIPL>>>>
        vipl_paths = sorted(set(
            glob.glob(os.path.join(data_path, "p*_v*")) +
            glob.glob(os.path.join(data_path, "VIPL", "p*_v*"))
        ))
        for vipl_dir in vipl_paths:
            frames_path = os.path.join(vipl_dir, "frames")
            basename = os.path.basename(vipl_dir)
            label_path = os.path.join(vipl_dir, f"{basename}.txt")
            time_path = os.path.join(vipl_dir, "time.txt")
            if os.path.isdir(frames_path) and os.path.exists(label_path):
                samples.append({
                    "index": basename,
                    "path": vipl_dir,
                    "dataset_type": "VIPL",
                    "label_path": label_path,
                    "time_path": time_path if os.path.exists(time_path) else None
                })
        if not samples:
            raise ValueError(self.dataset_name + " data paths empty!")
        return samples
    

    #*************************************************************************************************************
    # get_raw_data로 얻은 dict 리스트에서 begin, end 비율에 맞게 샘플링하여 반환 (예: 0.0-0.8이면 처음 80% 샘플 반환)
    # 학습/ 검증용으로 나누는 함수 -> 데이터 분할
    # data_dirs는 list[dict] 형태

    def split_raw_data(self, data_dirs, begin, end):
        if begin == 0 and end == 1:
            return data_dirs

        split_dirs = []
        split_seed = 100
        dataset_types = sorted(set(item.get("dataset_type", "") for item in data_dirs))
        for dataset_type in dataset_types:
            dataset_items = sorted(
                [item for item in data_dirs if item.get("dataset_type", "") == dataset_type],
                key=lambda item: item["index"])
            data_by_subject = {}
            for item in dataset_items:
                subject = self._get_subject_key(item)
                data_by_subject.setdefault(subject, []).append(item)

            subjects = sorted(data_by_subject.keys())
            rng = np.random.default_rng(split_seed)
            rng.shuffle(subjects)

            num_subjects = len(subjects)
            start_idx = int(begin * num_subjects)
            end_idx = int(end * num_subjects)
            selected_subjects = set(subjects[start_idx:end_idx])

            for item in dataset_items:
                if self._get_subject_key(item) in selected_subjects:
                    split_dirs.append(item)
        return split_dirs

    @staticmethod
    def _get_subject_key(data_dir):
        dataset_type = data_dir.get("dataset_type", "")
        index = data_dir["index"]
        if dataset_type == "UBFC":
            return index
        if dataset_type == "PURE":
            return index.split("-")[0]
        if dataset_type == "VIPL":
            match = re.match(r"(p\d+)_v\d+", index)
            return match.group(1) if match else index
        return index


    #*************************************************************************************************************
    # i번째 샘플을 읽음 -> 프레임/라벨 전처리 -> 클립 단위로 저장 -> 저장된 클립 이름 딕셔너리 리스트 반환 -> file_list_dict[i]에 저장
    # 매 epoch마다 프레임을 읽고 필터링 하면 너무 느리기 때문에 학습을 빠르게 만들기 위한 오프라인 전처리 함수 -> 멀티프로세싱으로 병렬 처리 가능

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        data_dir = data_dirs[i]
        dataset_type = data_dir["dataset_type"]
        # filename = os.path.basename(data_dir["path"])                                         # 일단 안쓰기 때문에 주석처리
        saved_filename = data_dir["index"]                                                      # 저장할 때 쓸 파일 이름

        if 'Motion' in config_preprocess.DATA_AUG:
            frames = self.read_npy_video(glob.glob(os.path.join(data_dir["path"], '*.npy')))    # npy 파일로 프레임 읽기
            sample_fs = getattr(self.config_data, 'FS', 30.0)                                   # 샘플 주파수
            time_values = None
        else:
            frame_dir = os.path.join(data_dir["path"], "frames")                                # frame 파일에서 이미지들을 읽는 함수         
            frames = self.read_video(frame_dir)

            if dataset_type == "PURE":                                  # 데이터셋이 PURE이면 30Hz로 고정
                sample_fs = 30.0
            elif dataset_type == "VIPL":
                time_values = self.read_wave(data_dir["time_path"]) if data_dir.get("time_path") else None
                sample_fs = self._estimate_fs_from_time(time_values)
                if not sample_fs or sample_fs <= 0:
                    sample_fs = getattr(self.config_data, 'FS', 30.0) or 30.0
            else:                                                       # UBFC이면 원본 fps 읽어서 사용 (읽기 실패하면 30Hz로 fallback)
                subj = data_dir["index"]  # "subject1"
                sample_fs = self._resolve_ubfc_fps(subj)                # fps 읽기
                if not sample_fs or sample_fs <= 0:
                    sample_fs = 30.0                                    # 마지막 fallback (원하면 에러로 강제해도 됨)
                time_values = None

        # 가짜 PPG 라벨을 생성하는 옵션 
        if config_preprocess.USE_PSUEDO_PPG_LABEL:                      # 라벨이 없거나 품질이 낮을 때, self-supervised 방식으로 학습/ 전처리
            bvps = self.generate_pos_psuedo_labels(frames, fs=sample_fs)
        # 실제 라벨 파일명이 *_waveform_30hz.txt 패턴. 
        # 없으면 바로 중단 => 라벨 없는 샘플은 학습 못함
        else:
            wave_file = data_dir["label_path"]
            if not wave_file or not os.path.exists(wave_file):
                raise FileNotFoundError(f"Label file not found for {data_dir['path']}")
            bvps = self.read_wave(wave_file)                            # self.read_wave 변환 타입: 보통 np.ndarray

        if dataset_type == "VIPL" and 'Motion' not in config_preprocess.DATA_AUG:
            frames, bvps, sample_fs = self._align_vipl_frames_and_labels(
                frames, bvps, time_values, getattr(self.config_data, 'FS', 30.0) or 30.0)
        else:
            min_len = min(len(frames), len(bvps))

            if len(frames) != len(bvps):
                print(f"[Warning] {dataset_type} length mismatch in {saved_filename}: "
                      f"frames={len(frames)}, bvps={len(bvps)}. Cropping to {min_len}")

            frames = frames[:min_len]
            bvps = bvps[:min_len]

        if dataset_type == "VIPL":
            frames_clips, bvps_clips = self.preprocess_vipl_roi(
                frames, bvps, config_preprocess, fs=sample_fs)
        else:
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess, fs=sample_fs)       # 전처리하여 클립으로 자르기
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
        return np.stack(frames, axis=0)

    def preprocess_vipl_roi(self, frames, bvps, config_preprocess, fs=None):
        """Preprocess VIPL frames without a second face crop."""
        frames = self.resize_with_letterbox(
            frames,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H)

        data = list()
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)

        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        chunk_length = config_preprocess.CHUNK_LENGTH
        chunk_length_seconds = getattr(config_preprocess, 'CHUNK_LENGTH_SEC', 0.0)
        if chunk_length_seconds and fs and fs > 0:
            frames_per_window = int(round(float(chunk_length_seconds) * fs))
            if frames_per_window > 0:
                chunk_length = frames_per_window
        chunk_length = max(1, chunk_length)

        if config_preprocess.DO_CHUNK:
            return self.chunk(data, bvps, chunk_length)
        return np.array([data]), np.array([bvps])

    @staticmethod
    def resize_with_letterbox(frames, width, height):
        """Resize ROI frames with preserved aspect ratio and edge padding."""
        total_frames, _, _, channels = frames.shape
        resized_frames = np.zeros((total_frames, height, width, channels), dtype=frames.dtype)
        for i, frame in enumerate(frames):
            src_h, src_w = frame.shape[:2]
            scale = min(width / src_w, height / src_h)
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pad_left = (width - new_w) // 2
            pad_right = width - new_w - pad_left
            pad_top = (height - new_h) // 2
            pad_bottom = height - new_h - pad_top
            resized_frames[i] = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        return resized_frames


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

    @staticmethod
    def _estimate_fs_from_time(time_values):
        if time_values is None or len(time_values) < 2:
            return 0.0
        diffs = np.diff(np.asarray(time_values, dtype=np.float64))
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return 0.0
        return float(1000.0 / np.median(diffs))

    @staticmethod
    def _align_vipl_frames_and_labels(frames, bvps, time_values, target_fs):
        if len(frames) == len(bvps) and (
                time_values is None or len(time_values) != len(frames) or target_fs <= 0):
            return frames, bvps, target_fs

        if time_values is None or len(time_values) != len(frames) or len(time_values) != len(bvps):
            target_length = min(len(frames), len(bvps))
            if len(frames) != len(bvps):
                print(f"[Warning] VIPL length mismatch without usable time.txt: "
                      f"frames={len(frames)}, bvps={len(bvps)}. Cropping to {target_length}")
            return frames[:target_length], bvps[:target_length], target_fs

        time_values = np.asarray(time_values, dtype=np.float64)
        bvps = np.asarray(bvps, dtype=np.float64)
        if target_fs <= 0 or len(time_values) < 2:
            return frames, bvps, target_fs

        start_time = time_values[0]
        end_time = time_values[-1]
        if end_time <= start_time:
            return frames, bvps, target_fs

        step_ms = 1000.0 / float(target_fs)
        target_times = np.arange(start_time, end_time + 0.5 * step_ms, step_ms)
        if len(target_times) < 1:
            return frames, bvps, target_fs

        nearest = np.searchsorted(time_values, target_times, side="left")
        nearest = np.clip(nearest, 0, len(time_values) - 1)
        prev_idx = np.maximum(nearest - 1, 0)
        use_prev = np.abs(time_values[prev_idx] - target_times) < np.abs(time_values[nearest] - target_times)
        nearest[use_prev] = prev_idx[use_prev]

        frames = frames[nearest]
        bvps = np.interp(target_times, time_values, bvps)
        return frames, bvps, float(target_fs)


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
