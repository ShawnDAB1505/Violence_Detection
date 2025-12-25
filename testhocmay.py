import os
import cv2
import numpy as np
import joblib
import gc
import hashlib
import time
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from xgboost import XGBClassifier

# Sklearn & Ensemble
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.svm import LinearSVC    
from sklearn.calibration import CalibratedClassifierCV


# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'TRAIN_DIR': r"C:\Users\admin\Desktop\Xla\Train Dataset\Train Dataset",
    'TEST_DIR': r"C:\Users\admin\Desktop\Xla\Train Dataset\test video", 
    
    # Folder mới để lưu dữ liệu đã tính toán (Cache)
    'CACHE_DIR': r"C:\Users\admin\Desktop\Xla\extracted_features",
    'MODEL_SAVE_PATH': r"C:\Users\admin\Desktop\Xla\violence_ensemble_model_final.joblib",
    
    'IMG_SIZE': (64, 64),  
    'FRAMES_PER_VIDEO': 60,
    
    # HOG Params
    'HOG_WIN_SIZE': (64, 64),
    'HOG_BLOCK_SIZE': (16, 16),
    'HOG_BLOCK_STRIDE': (8, 8),
    'HOG_CELL_SIZE': (8, 8),
    'HOG_NBINS': 9 
}


# Tạo thư mục cache nếu chưa có
os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)

# ==========================================
# 1. FEATURE EXTRACTION (V5 - WITH ACCELERATION)
# ==========================================
def extract_combined_features(prev_frame, curr_frame, prev_mag=None, prev_flow=None):
    """
    PHIÊN BẢN V5 (Sửa lỗi & Tối ưu):
    - Thêm tham số prev_mag để tính Acceleration.
    - Trích xuất 8 chỉ số motion để phân biệt Thể thao/Bạo lực.
    """
    if prev_frame is None or curr_frame is None:
        return None, None, None

    # 1. Tiền xử lý (Resize & Gray)
    prev_gray = cv2.cvtColor(cv2.resize(prev_frame, CONFIG['IMG_SIZE']), cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(cv2.resize(curr_frame, CONFIG['IMG_SIZE']), cv2.COLOR_BGR2GRAY)

    # 2. HOG Features
    hog = cv2.HOGDescriptor(
        CONFIG['HOG_WIN_SIZE'], CONFIG['HOG_BLOCK_SIZE'], 
        CONFIG['HOG_BLOCK_STRIDE'], CONFIG['HOG_CELL_SIZE'], 
        CONFIG['HOG_NBINS']
    )
    hog_feats = hog.compute(curr_gray)
    if hog_feats is None: return None, None
    hog_feats = hog_feats.flatten()

    # 3. Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Lọc nhiễu: mag > 2.5 (giảm từ 4.0 để bắt violence nhẹ hơn)
    valid_mask = mag > 1.2
    
    # Khởi tạo các giá trị mặc định
    mean_mag = std_mag = max_mag = motion_area = angle_std = symmetry = 0.0
    acceleration = accel_jitter = 0.0
    flow_consistency = directional_stability = rhythm_score = edge_density = 0.0

    if np.sum(valid_mask) > 10:
        valid_mag = mag[valid_mask]
        valid_ang = ang[valid_mask]
        
        # Chỉ số Motion cơ bản
        mean_mag = np.mean(valid_mag)
        std_mag = np.std(valid_mag)
        max_mag = np.max(valid_mag)
        motion_area = np.sum(valid_mask) / mag.size
        angle_std = np.std(valid_ang)
        
        # Gia tốc & Độ rung (Jerk/Jitter)
        if prev_mag is not None:
            accel_map = mag - prev_mag
            valid_accel = accel_map[valid_mask]
            acceleration = np.mean(np.abs(valid_accel))  # Lấy abs để tránh âm
            accel_jitter = np.std(valid_accel)  # Độ hỗn loạn của gia tốc

        # Tính đối xứng (Symmetry)
        h, w = mag.shape
        left = np.sum(mag[:, :w//2])
        right = np.sum(mag[:, w//2:])
        symmetry = abs(left - right) / (left + right + 1e-5)
    # 1. FLOW CONSISTENCY (Temporal Coherence)
        # Đo sự thay đổi hướng giữa 2 frame → Sports nhỏ, Violence lớn
        if prev_flow is not None:
            flow_diff = np.abs(flow - prev_flow)
            flow_consistency = np.mean(flow_diff[valid_mask]) if np.sum(valid_mask) > 10 else 0.0
        
        # 2. DIRECTIONAL STABILITY (Chạy/Lướt ván có hướng ổn định)
        # Tính độ tập trung của góc chuyển động
        angle_bins = np.histogram(valid_ang, bins=8, range=(0, 2*np.pi))[0]
        angle_entropy = -np.sum((angle_bins / np.sum(angle_bins)) * 
                                np.log(angle_bins / np.sum(angle_bins) + 1e-10))
        directional_stability = 1.0 / (angle_entropy + 1e-5)  # Càng tập trung → càng cao
        
        # 3. RHYTHM SCORE (Phát hiện chuyển động nhịp nhàng)
        # Dùng FFT để phát hiện periodicity trong magnitude
        if len(valid_mag) > 16:
            fft = np.fft.fft(valid_mag[:256] if len(valid_mag) > 256 else valid_mag)
            power = np.abs(fft[:len(fft)//2])
            rhythm_score = np.max(power[1:]) / (np.mean(power) + 1e-5)  # Peak/Mean ratio
        
        # 4. EDGE DENSITY (Phát hiện va chạm/đánh nhau)
        edges = cv2.Canny(curr_gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size


    motion_feats = np.array([
        mean_mag, acceleration, accel_jitter, std_mag, 
        max_mag, motion_area, angle_std, symmetry, flow_consistency, directional_stability, rhythm_score, edge_density
    ])

    return np.hstack([hog_feats, motion_feats]), mag, flow

# ==========================================
# 2. SMART CACHING SYSTEM (Tăng tốc Train)
# ==========================================
def get_cache_path(video_path):
    video_id = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    return os.path.join(CONFIG['CACHE_DIR'], os.path.basename(video_path) + f"_{video_id}.joblib")

def process_single_video_train(video_path, label_idx):
    """
    ===== ĐÃ SỬA: NHẬN 2 GIÁ TRỊ TỪ extract_combined_features =====
    """
    
    video_name = os.path.basename(video_path)
    worker_id = os.getpid() 
    print(f"[CPU-{worker_id}] Dang xu ly: {video_name}")

    cache_path = get_cache_path(video_path)
    
    # LOAD CACHE
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    
    # TÍNH TOÁN
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    X_temp, y_temp = [], []
    
    if total_frames > 0:
        step = max(1, total_frames // CONFIG['FRAMES_PER_VIDEO'])
        count = 0
        prev_frame = None
        prev_mag = None  
        prev_flow = None
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if count % step == 0:
                if prev_frame is not None:
                    # ===== SỬA: NHẬN 2 GIÁ TRỊ =====
                    feat, curr_mag, curr_flow = extract_combined_features(prev_frame, frame, prev_mag)
                    
                    if feat is not None:
                        # 1. Gốc
                        X_temp.append(feat)  # ← Bây giờ feat là array, không phải tuple!
                        y_temp.append(label_idx)
                        
                        # 2. Data Augmentation (Flip)
                        feat_flip, _, _ = extract_combined_features(
                            cv2.flip(prev_frame, 1), 
                            cv2.flip(frame, 1),
                            cv2.flip(prev_mag, 1) if prev_mag is not None else None,
                            None
                        )
                        if feat_flip is not None:
                            X_temp.append(feat_flip)
                            y_temp.append(label_idx)
                    
                    # ===== CẬP NHẬT prev_mag =====
                    prev_mag = curr_mag
                    prev_flow = curr_flow

                prev_frame = frame
            count += 1
    cap.release()
    
    result = (np.array(X_temp), np.array(y_temp))
    
    # Lưu Cache
    if len(X_temp) > 0:
        joblib.dump(result, cache_path)
        
    return result


def load_dataset_parallel(data_dir):
    print(f"[INFO] Scanning dataset & using CACHE at {CONFIG['CACHE_DIR']}...")
    classes = {'non_violence': 0, 'violence': 1}
    video_tasks = []
    for label_name, label_idx in classes.items():
            folder_path = os.path.join(data_dir, label_name)
            if not os.path.exists(folder_path): continue
            files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                files.extend(glob(os.path.join(folder_path, ext)))
            for f in files: video_tasks.append((f, label_idx))
                
    print(f"[INFO] Processing {len(video_tasks)} videos using Parallel CPU...")
    # Chạy đa luồng
    results = Parallel(n_jobs=-1)(delayed(process_single_video_train)(p, l) for p, l in tqdm(video_tasks))
        
    X_list, y_list = [], []
    for X_v, y_v in results:
        if len(X_v) > 0:
            # Chuyển sang float32 ngay để tiết kiệm RAM mà không mất độ chính xác
            X_list.append(X_v.astype(np.float32))
            y_list.append(y_v.astype(np.int8))
    
    # Giải phóng results ngay lập tức
    del results
    gc.collect()

    # Ghép mảng một cách hiệu quả
    X_final = np.vstack(X_list)
    y_final = np.concatenate(y_list)
    
    del X_list, y_list
    gc.collect()
    
    return X_final, y_final

# ==========================================
# 3. MODEL PIPELINE (XGBoost Ensemble)
# ==========================================
def build_ensemble_pipeline():
    # SVM
    svm_clf = LinearSVC(C=0.3, class_weight='balanced', random_state=42, dual=False, max_iter=1500)
    svm_calibrated = CalibratedClassifierCV(svm_clf, cv=3)

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_leaf=4, 
        class_weight='balanced_subsample', n_jobs=2, random_state=42
    )
    
    # XGBoost
    xgb_clf = XGBClassifier(
        n_estimators=200, 
        max_depth=8,
        learning_rate=0.06,
        min_child_weight=2, 
        grow_policy='lossguide',
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.5,
        device='cuda', 
        tree_method='hist',
        predictor='gpu_predictor',
        scale_pos_weight=1.2,
        eval_metric='logloss',
        random_state=42,
        n_jobs=1
    )
    
    ensemble = VotingClassifier(
        estimators=[('svm', svm_calibrated), ('rf', rf_clf), ('xgb', xgb_clf)],
        voting='soft',
        weights=[1.2, 1, 2.6],
        n_jobs=1
    )
    
    return ImbPipeline([('scaler', StandardScaler()), ('classifier', ensemble)])

# ==========================================
# 4. SMART INFERENCE (WITH ACCELERATION LOGIC)
# ==========================================
def is_video_violent_smart(video_path, model, threshold=0.9):
    """
    ULTIMATE VERSION - Hockey-Aware Detection
    
    Key Improvements:
    1. Separate "Contact Sports" from "Clean Sports" detection
    2. Chaos-based violence detection (works for hockey fights)
    3. Model confidence gating (prevent false penalties)
    4. Multi-factor decision tree
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        return False
    
    prev_frame = None
    prev_mag = None
    prev_flow = None
    frame_skip = 4
    cnt = 0
    
    smoothed_score = 0
    decay = 0.85
    max_score = 0
    max_consecutive = 0
    consecutive_frames = 0
    
    # Enhanced tracking
    total_frames = 0
    frames_with_motion = 0
    frames_with_strong_motion = 0
    frames_with_weak_motion = 0
    static_frames = 0
    
    high_prob_frames = 0
    high_raw_prob_frames = 0
    
    # Pattern tracking
    chaos_frames = 0
    impact_frames = 0
    clean_sports_frames = 0  # Renamed from sports_frames
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if cnt % frame_skip == 0:
            if prev_frame is not None:
                feat, curr_mag, curr_flow = extract_combined_features(prev_frame, frame, prev_mag)
                
                if feat is not None:
                    total_frames += 1
                    motion_stats = feat[-12:]
                    
                    mean_mag = motion_stats[0]
                    acceleration = motion_stats[1]
                    accel_jitter = motion_stats[2]
                    std_mag = motion_stats[3]
                    max_mag = motion_stats[4]
                    motion_area = motion_stats[5]
                    angle_std = motion_stats[6]
                    symmetry = motion_stats[7]
                    flow_consistency = motion_stats[8]
                    directional_stability = motion_stats[9]
                    rhythm_score = motion_stats[10]
                    edge_density = motion_stats[11]
                    # Classify frame motion type
                    if mean_mag < 0.8:
                        static_frames += 1
                    elif mean_mag < 2.0:
                        frames_with_weak_motion += 1
                        frames_with_motion += 1
                    elif mean_mag < 4.0:
                        frames_with_motion += 1
                    else:
                        frames_with_strong_motion += 1
                        frames_with_motion += 1
                    
                    # Pattern classification (refined)
                    if accel_jitter > 3.0:
                        chaos_frames += 1
                    
                    if mean_mag > 3.0 and accel_jitter < 1.5:
                        impact_frames += 1
                    
                    # CRITICAL: Only count CLEAN sports (not contact sports like hockey)
                    if mean_mag > 5.5 and angle_std < 0.6 and accel_jitter < 1.5:
                        clean_sports_frames += 1
                    
                    feat = feat.reshape(1, -1)
                    raw_prob = model.predict_proba(feat)[0][1]
                    
                    if raw_prob > 0.65:
                        high_raw_prob_frames += 1
                    
                    # ===============================================
                    # CLASSIFICATION LOGIC - CONTEXT-AWARE
                    # ===============================================
                    
                    # TIER 1: STATIC DETECTION (Unchanged - working well)
                    if mean_mag < 0.5:
                        if raw_prob > 0.90:
                            final_prob = raw_prob * 0.3
                        else:
                            final_prob = raw_prob * 0.1
                    
                    elif mean_mag < 0.8:
                        if raw_prob > 0.85:
                            final_prob = raw_prob * 0.5
                        elif raw_prob > 0.7:
                            final_prob = raw_prob * 0.3
                        else:
                            final_prob = raw_prob * 0.15
                    
                    # ===============================================
                    # TIER 2: VIOLENCE DETECTION (Priority over sports)
                    # ===============================================
                    
                    # Pattern A: EXTREME CHAOS (Hockey fights, brawls)
                    # High jitter + High std_mag = Erratic violence
                    elif accel_jitter > 4.0 and std_mag > 3.0:
                        final_prob = min(raw_prob * 1.6, 1.0)
                    
                    # Pattern B: HIGH CHAOS with moderate motion
                    # Typical hockey fight signature
                    elif accel_jitter > 3.5 and mean_mag > 2.0:
                        final_prob = min(raw_prob * 1.5, 1.0)
                    
                    # Pattern C: SUSTAINED CHAOS (not just one spike)
                    # Multiple erratic movements
                    elif accel_jitter > 3.0 and angle_std > 0.9:
                        final_prob = min(raw_prob * 1.4, 1.0)
                    
                    # Pattern D: MID-RANGE ERRATIC (shoving, hitting)
                    # Medium motion + high chaos + model confident
                    elif (mean_mag > 2.0 and mean_mag < 4.5 and 
                          accel_jitter > 2.5 and angle_std > 0.8):
                        final_prob = min(raw_prob * 1.3, 1.0)
                    # Pattern E: EDGE-HEAVY CHAOS (Đánh đấm có va chạm)
                    # Đặc điểm: Edge nhiều + Chaos cao + Model confident
                    elif (edge_density > 0.15 and accel_jitter > 2.8 and raw_prob > 0.70):
                        final_prob = min(raw_prob * 1.35, 1.0)
                    
                    # Pattern F: ERRATIC DIRECTION CHANGE (Hỗn loạn hướng)
                    # Đặc điểm: Flow không ổn định + High jitter
                    elif (flow_consistency > 3.0 and accel_jitter > 2.5):
                        final_prob = min(raw_prob * 1.25, 1.0)

                
                    # ===============================================
                    # TIER 3: CLEAN SPORTS DETECTION
                    # ===============================================
                    
                    # Pattern A: RHYTHMIC FAST MOTION (Chạy bộ, đạp xe)
                    # Đặc điểm: Nhanh + Nhịp nhàng + Ổn định hướng + Low chaos
                    elif (mean_mag > 4.0 and rhythm_score > 2.5 and 
                          directional_stability > 0.8 and accel_jitter < 2.0):
                        final_prob = raw_prob * 0.02  # Penalty rất mạnh
                    
                    # Pattern B: SMOOTH DIRECTIONAL (Lướt ván, trượt tuyết)
                    # Đặc điểm: Nhanh + Hướng tập trung + Flow ổn định
                    elif (mean_mag > 5.0 and directional_stability > 1.0 and 
                          flow_consistency < 1.5 and raw_prob < 0.65):
                        final_prob = raw_prob * 0.03
                    
                    # Pattern C: PURE ORGANIZED SPORTS (Bóng đá, bóng rổ)
                    # Đặc điểm: Rất nhanh + Smooth + Góc ổn định + Model không confident
                    elif (mean_mag > 6.5 and angle_std < 0.55 and 
                          accel_jitter < 1.8 and raw_prob < 0.60):
                        final_prob = raw_prob * 0.03
                    
                    # Pattern D: FAST SYMMETRIC (Chạy đường dài, bơi lội)
                    # Đặc điểm: Nhanh + Cân bằng + Nhịp nhàng
                    elif (mean_mag > 6.0 and symmetry < 0.3 and 
                          rhythm_score > 2.0 and raw_prob < 0.65):
                        final_prob = raw_prob * 0.04
                    
                    # Pattern E: MEDIUM-HIGH ORGANIZED (Tennis, Volleyball)
                    elif (mean_mag > 5.0 and angle_std < 0.60 and 
                          accel_jitter < 2.0 and raw_prob < 0.70):
                        final_prob = raw_prob * 0.08
                    
                    # Pattern F: SMOOTH CONTINUOUS (Cycling, Swimming)
                    elif (mean_mag > 4.5 and std_mag < 1.5 and 
                          accel_jitter < 1.5 and flow_consistency < 2.0):
                        final_prob = raw_prob * 0.10
                    
                    # ===============================================
                    # TIER 4: CONTACT SPORTS / AMBIGUOUS ZONE
                    # ===============================================
                    
                    # HIGH MOTION + MODERATE CHAOS + MODEL NOT CONFIDENT
                    # This is where hockey gameplay (not fights) falls
                    # Only penalty if model is NOT confident
                    elif mean_mag > 4.5 and accel_jitter < 3.0 and raw_prob < 0.65:
                        final_prob = raw_prob * 0.30  # Moderate penalty
                    
                    # MEDIUM-HIGH MOTION + LOW CHAOS + LOW CONFIDENCE
                    # Normal sports activity
                    elif mean_mag > 3.5 and accel_jitter < 2.5 and raw_prob < 0.70:
                        final_prob = raw_prob * 0.50
                    
                    # MEDIUM MOTION + LOW CHAOS
                    # Walking, jogging, normal activities
                    elif mean_mag > 2.5 and accel_jitter < 2.0:
                        final_prob = raw_prob * 0.65
                    
                    # LOW-MEDIUM MOTION
                    elif mean_mag > 1.5 and mean_mag < 2.5:
                        final_prob = raw_prob * 0.75
                    
                    # ===============================================
                    # TIER 5: DEFAULT (Trust model)
                    # ===============================================
                    else:
                        final_prob = raw_prob
                    
                    max_score = max(max_score, final_prob)
                    
                    if final_prob > 0.65:
                        high_prob_frames += 1
                    
                    # EMA Smoothing
                    if smoothed_score == 0: 
                        smoothed_score = final_prob
                    else: 
                        smoothed_score = decay * smoothed_score + (1 - decay) * final_prob
                    
                    # Consecutive counting - only with meaningful motion
                    if smoothed_score > 0.55 and mean_mag > 0.8:
                        consecutive_frames += 1
                    else: 
                        consecutive_frames = 0
                    max_consecutive = max(max_consecutive, consecutive_frames)
                
                prev_mag = curr_mag
                prev_flow = curr_flow
            
            prev_frame = frame
        cnt += 1
    
    cap.release()
    
    if total_frames == 0:
        return False
    
    # ===============================================
    # DECISION LOGIC - MULTI-FACTOR TREE
    # ===============================================
    
    # Calculate ratios
    static_ratio = static_frames / total_frames
    weak_motion_ratio = frames_with_weak_motion / total_frames
    motion_ratio = frames_with_motion / total_frames
    strong_motion_ratio = frames_with_strong_motion / total_frames
    
    high_prob_ratio = high_prob_frames / total_frames
    high_raw_prob_ratio = high_raw_prob_frames / total_frames
    
    chaos_ratio = chaos_frames / total_frames
    impact_ratio = impact_frames / total_frames
    clean_sports_ratio = clean_sports_frames / total_frames
    
    # ===============================================
    # DECISION TREE
    # ===============================================
    
    # Branch 1: STATIC VIDEO (>70% static)
    if static_ratio > 0.70:
        if high_raw_prob_ratio > 0.60 and max_score > 0.90:
            return True
        return False
    
    # Branch 2: MOSTLY STATIC (50-70%)
    elif static_ratio > 0.50:
        return (
            (max_consecutive >= 10 and smoothed_score > 0.80) or
            (max_score > 0.95 and high_prob_ratio > 0.30)
        )
    
    # Branch 3: CHAOS-DOMINATED (>25% chaotic - LOWERED from 30%)
    # This is KEY for hockey violence detection
    elif chaos_ratio > 0.25:
        return (
            (max_consecutive >= 4) or  # Lowered from 5
            (smoothed_score > 0.60) or  # Lowered from 0.65
            (max_score > 0.75) or  # Lowered from 0.80
            (high_prob_ratio > 0.20 and max_score > 0.65)
        )
    
    # Branch 4: CLEAN SPORTS (>40% clean sports frames)
    # Only applies to non-contact sports
    elif clean_sports_ratio > 0.40 or (strong_motion_ratio > 0.50 and chaos_ratio < 0.15):
        return (
            (max_consecutive >= 10 and smoothed_score > 0.90) or
            (smoothed_score > 0.90 and max_score > 0.98) or
            (chaos_ratio > 0.35 and max_score > 0.95)  # Exception: chaotic override
        )
    
    # Branch 5: HIGH MOTION but not clean sports (Contact sports zone)
    # Hockey gameplay falls here
    elif strong_motion_ratio > 0.35:
        # More lenient thresholds - trust model more
        return (
            (max_consecutive >= 6) or
            (smoothed_score > 0.75) or
            (max_score > 0.88) or
            (chaos_ratio > 0.20 and max_score > 0.80)  # Chaotic boost
        )
    
    # Branch 6: NORMAL MOTION
    elif motion_ratio > 0.20:
        # SAFEGUARD: If high max_score but low smoothed/consecutive, be cautious
        if max_score > 0.88 and smoothed_score < 0.60 and max_consecutive < 3:
            # Likely a spike, not sustained violence
            return False
        
        return (
            (max_consecutive >= 6) or
            (smoothed_score > 0.75) or
            (max_score > 0.92) or  # Raised from 0.88 to 0.92
            (high_prob_ratio > 0.35 and max_score > 0.85)
        )
    
    # Branch 7: LOW MOTION
    elif motion_ratio > 0.10:
        return (
            (max_consecutive >= 5) or
            (smoothed_score > 0.70) or
            (max_score > 0.85) or
            (high_raw_prob_ratio > 0.40 and max_score > 0.75)
        )
    
    # Branch 8: MINIMAL MOTION
    else:
        if high_raw_prob_ratio > 0.50 and max_score > 0.80:
            return True
        return False

def is_video_violent_smart_debug(video_path, model, threshold=0.9):
    """Debug version with hockey-specific logging"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        return False
    
    prev_frame = None
    prev_mag = None
    prev_flow = None
    frame_skip = 4
    cnt = 0
    
    smoothed_score = 0
    decay = 0.85
    max_score = 0
    max_consecutive = 0
    consecutive_frames = 0
    
    total_frames = 0
    frames_with_motion = 0
    frames_with_strong_motion = 0
    frames_with_weak_motion = 0
    static_frames = 0
    
    high_prob_frames = 0
    high_raw_prob_frames = 0
    
    chaos_frames = 0
    impact_frames = 0
    clean_sports_frames = 0
    
    debug_info = {
        'max_mean_mag': 0,
        'max_jitter': 0,
        'static_kill': 0,
        'violence_boost': 0,
        'clean_sports_penalty': 0,
        'contact_sports_penalty': 0,
        'raw_probs': []
    }
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if cnt % frame_skip == 0:
            if prev_frame is not None:
                feat, curr_mag, curr_flow = extract_combined_features(prev_frame, frame, prev_mag, None)
                
                if feat is not None:
                    total_frames += 1
                    motion_stats = feat[-12:]
                    
                    mean_mag = motion_stats[0]
                    accel_jitter = motion_stats[2]
                    std_mag = motion_stats[3]
                    angle_std = motion_stats[6]
                    symmetry = motion_stats[7]
                    flow_consistency = motion_stats[8]
                    directional_stability = motion_stats[9]
                    rhythm_score = motion_stats[10]
                    edge_density = motion_stats[11]
                    if mean_mag < 0.8:
                        static_frames += 1
                    elif mean_mag < 2.0:
                        frames_with_weak_motion += 1
                        frames_with_motion += 1
                    elif mean_mag < 4.0:
                        frames_with_motion += 1
                    else:
                        frames_with_strong_motion += 1
                        frames_with_motion += 1
                    
                    if accel_jitter > 3.0:
                        chaos_frames += 1
                    if mean_mag > 3.0 and accel_jitter < 1.5:
                        impact_frames += 1
                    if mean_mag > 5.5 and angle_std < 0.6 and accel_jitter < 1.5:
                        clean_sports_frames += 1
                    
                    debug_info['max_mean_mag'] = max(debug_info['max_mean_mag'], mean_mag)
                    debug_info['max_jitter'] = max(debug_info['max_jitter'], accel_jitter)
                    
                    feat = feat.reshape(1, -1)
                    raw_prob = model.predict_proba(feat)[0][1]
                    debug_info['raw_probs'].append(raw_prob)
                    
                    if raw_prob > 0.65:
                        high_raw_prob_frames += 1
                    
                    # Classification with tracking
                    if mean_mag < 0.5:
                        debug_info['static_kill'] += 1
                        final_prob = raw_prob * (0.3 if raw_prob > 0.90 else 0.1)
                    
                    elif mean_mag < 0.8:
                        debug_info['static_kill'] += 1
                        if raw_prob > 0.85:
                            final_prob = raw_prob * 0.5
                        elif raw_prob > 0.7:
                            final_prob = raw_prob * 0.3
                        else:
                            final_prob = raw_prob * 0.15
                    
                    # VIOLENCE PATTERNS
                    elif accel_jitter > 4.0 and std_mag > 3.0:
                        debug_info['violence_boost'] += 1
                        final_prob = min(raw_prob * 1.6, 1.0)
                    
                    elif accel_jitter > 3.5 and mean_mag > 2.0:
                        debug_info['violence_boost'] += 1
                        final_prob = min(raw_prob * 1.5, 1.0)
                    
                    elif accel_jitter > 3.0 and angle_std > 0.9:
                        debug_info['violence_boost'] += 1
                        final_prob = min(raw_prob * 1.4, 1.0)
                    
                    elif (mean_mag > 2.0 and mean_mag < 4.5 and 
                          accel_jitter > 2.5 and angle_std > 0.8):
                        debug_info['violence_boost'] += 1
                        final_prob = min(raw_prob * 1.3, 1.0)
                    
                    # CLEAN SPORTS PATTERNS
                    elif (mean_mag > 6.5 and angle_std < 0.55 and 
                          accel_jitter < 1.8 and raw_prob < 0.60):
                        debug_info['clean_sports_penalty'] += 1
                        final_prob = raw_prob * 0.03
                    
                    elif (mean_mag > 6.0 and symmetry < 0.3 and 
                          accel_jitter < 2.0 and raw_prob < 0.65):
                        debug_info['clean_sports_penalty'] += 1
                        final_prob = raw_prob * 0.05
                    
                    elif (mean_mag > 5.0 and angle_std < 0.60 and 
                          accel_jitter < 2.0 and raw_prob < 0.70):
                        debug_info['clean_sports_penalty'] += 1
                        final_prob = raw_prob * 0.08
                    
                    elif (mean_mag > 4.5 and std_mag < 1.5 and 
                          accel_jitter < 1.5 and raw_prob < 0.70):
                        debug_info['clean_sports_penalty'] += 1
                        final_prob = raw_prob * 0.12
                    
                    # CONTACT SPORTS / AMBIGUOUS
                    elif mean_mag > 4.5 and accel_jitter < 3.0 and raw_prob < 0.65:
                        debug_info['contact_sports_penalty'] += 1
                        final_prob = raw_prob * 0.30
                    
                    elif mean_mag > 3.5 and accel_jitter < 2.5 and raw_prob < 0.70:
                        final_prob = raw_prob * 0.50
                    
                    elif mean_mag > 2.5 and accel_jitter < 2.0:
                        final_prob = raw_prob * 0.65
                    
                    elif mean_mag > 1.5 and mean_mag < 2.5:
                        final_prob = raw_prob * 0.75
                    
                    else:
                        final_prob = raw_prob
                    
                    max_score = max(max_score, final_prob)
                    
                    if final_prob > 0.65:
                        high_prob_frames += 1
                    
                    if smoothed_score == 0: 
                        smoothed_score = final_prob
                    else: 
                        smoothed_score = decay * smoothed_score + (1 - decay) * final_prob
                    
                    if smoothed_score > 0.55 and mean_mag > 0.8:
                        consecutive_frames += 1
                    else: 
                        consecutive_frames = 0
                    max_consecutive = max(max_consecutive, consecutive_frames)
                
                prev_mag = curr_mag
                prev_flow = curr_flow
            
            prev_frame = frame
        cnt += 1
    
    cap.release()
    
    if total_frames == 0:
        return False
    
    # Calculate ratios
    static_ratio = static_frames / total_frames
    motion_ratio = frames_with_motion / total_frames
    strong_motion_ratio = frames_with_strong_motion / total_frames
    high_prob_ratio = high_prob_frames / total_frames
    high_raw_prob_ratio = high_raw_prob_frames / total_frames
    chaos_ratio = chaos_frames / total_frames
    impact_ratio = impact_frames / total_frames
    clean_sports_ratio = clean_sports_frames / total_frames
    
    # DECISION LOGIC (matching main function)
    if static_ratio > 0.70:
        video_type = "STATIC"
        decision = high_raw_prob_ratio > 0.60 and max_score > 0.90
    
    elif static_ratio > 0.50:
        video_type = "MOSTLY_STATIC"
        decision = ((max_consecutive >= 10 and smoothed_score > 0.80) or
                   (max_score > 0.95 and high_prob_ratio > 0.30))
    
    elif chaos_ratio > 0.25:
        video_type = "CHAOS"
        decision = ((max_consecutive >= 4) or (smoothed_score > 0.60) or
                   (max_score > 0.75) or (high_prob_ratio > 0.20 and max_score > 0.65))
    
    elif clean_sports_ratio > 0.40 or (strong_motion_ratio > 0.50 and chaos_ratio < 0.15):
        video_type = "CLEAN_SPORTS"
        decision = ((max_consecutive >= 10 and smoothed_score > 0.90) or
                   (smoothed_score > 0.90 and max_score > 0.98) or
                   (chaos_ratio > 0.35 and max_score > 0.95))
    
    elif strong_motion_ratio > 0.35:
        video_type = "CONTACT_SPORTS"
        decision = ((max_consecutive >= 6) or (smoothed_score > 0.75) or
                   (max_score > 0.88) or (chaos_ratio > 0.20 and max_score > 0.80))
    
    elif motion_ratio > 0.20:
        video_type = "NORMAL"
        # SAFEGUARD for spikes
        if max_score > 0.88 and smoothed_score < 0.60 and max_consecutive < 3:
            decision = False
        else:
            decision = ((max_consecutive >= 6) or (smoothed_score > 0.75) or
                       (max_score > 0.92) or (high_prob_ratio > 0.35 and max_score > 0.85))
    
    elif motion_ratio > 0.10:
        video_type = "LOW_MOTION"
        decision = ((max_consecutive >= 5) or (smoothed_score > 0.70) or
                   (max_score > 0.85) or (high_raw_prob_ratio > 0.40 and max_score > 0.75))
    
    else:
        video_type = "MINIMAL"
        decision = high_raw_prob_ratio > 0.50 and max_score > 0.80
    
    # LOGGING
    video_name = os.path.basename(video_path)
    avg_prob = np.mean(debug_info['raw_probs']) if debug_info['raw_probs'] else 0
    
    print(f"\n[DEBUG] {video_name}")
    print(f"  Decision: {'VIOLENCE' if decision else 'SAFE'} | Type: {video_type}")
    print(f"  Motion: Static={static_ratio:.1%}, Normal={motion_ratio:.1%}, Strong={strong_motion_ratio:.1%}")
    print(f"  Pattern: Chaos={chaos_ratio:.1%}, Impact={impact_ratio:.1%}, CleanSports={clean_sports_ratio:.1%}")
    print(f"  Scores: Max={max_score:.3f}, Smooth={smoothed_score:.3f}, Consec={max_consecutive}")
    print(f"  Prob Ratios: High={high_prob_ratio:.1%}, RawHigh={high_raw_prob_ratio:.1%}, Avg={avg_prob:.3f}")
    print(f"  Features: Mag={debug_info['max_mean_mag']:.2f}, Jitter={debug_info['max_jitter']:.2f}")
    print(f"  Filters: Static={debug_info['static_kill']}, Violence={debug_info['violence_boost']}")
    print(f"           CleanSports={debug_info['clean_sports_penalty']}, ContactSports={debug_info['contact_sports_penalty']}")
    
    return decision


# Wrapper for parallel processing
def detect_wrapper(path, model):
    res = is_video_violent_smart_debug(path, model)
    return (os.path.basename(path), res)

def scan_folder_parallel(folder_path, model):
    video_paths = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_paths.extend(glob(os.path.join(folder_path, ext)))
        
    print(f"\n[INFO] Parallel Detecting {len(video_paths)} videos...")
    results = Parallel(n_jobs=-1)(delayed(detect_wrapper)(p, model) for p in tqdm(video_paths))
    return results




def format_runtime(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours} hours {minutes} minutes {secs} seconds"

# ==========================================
# MAIN EXECUTION (ĐÃ FIX LOGIC & RAM)
# ==========================================
if __name__ == "__main__":
    start_time = time.perf_counter()

    # 1. LOAD DATA (Nhanh nhờ Cache)
    X, y = load_dataset_parallel(CONFIG['TRAIN_DIR'])
    
    if len(X) == 0:
        print("[ERROR] No data. Check path.")
        exit()

    print(f"[INFO] Features Shape: {X.shape}")
    
    # 2. CHIA DATA & GIẢI PHÓNG RAM NGAY LẬP TỨC
    # Chuyển test_size về float32 để tiết kiệm bộ nhớ cho tập test
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y, test_size=0.2, stratify=y, random_state=42
    )
    
    # QUAN TRỌNG: Xóa X và y gốc ngay sau khi split để giải phóng RAM cho việc Train
    del X
    del y
    gc.collect() 

    # 3. KHỞI TẠO VÀ HUẤN LUYỆN
    print("[INFO] Training model...")
    model = build_ensemble_pipeline()
    
    # Sử dụng tqdm bao quanh quá trình fit (Lưu ý: fit có thể tốn RAM nhất)
    model.fit(X_train, y_train)
    
    # Sau khi fit xong, xóa X_train để lấy RAM cho việc Predict và Detect
    del X_train
    gc.collect()

    # 4. ĐÁNH GIÁ (Sử dụng X_test đã tách ra từ trước)
    y_pred = model.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    
    # Lưu mô hình
    joblib.dump(model, CONFIG['MODEL_SAVE_PATH'])
    
    # Xóa tiếp các biến không dùng tới trước khi Scan Folder
    del X_test, y_test, y_pred
    gc.collect()

    # 5. DETECT (Chạy song song)
    results = scan_folder_parallel(CONFIG['TEST_DIR'], model)
    
    violence = [n for n, r in results if r]
    non_violence = [n for n, r in results if not r]
    
    print("\n" + "="*30)
    print(f"Violence ({len(violence)}): {violence}")
    print(f"Non-Violence ({len(non_violence)}): {non_violence}")
    print("="*30)

    end_time = time.perf_counter()
    print("\n[INFO] Total Runtime:", format_runtime(end_time - start_time))