import os
import cv2
import numpy as np
import joblib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import time

# =====================================================
# ENHANCED DETECTOR WITH DUAL MODE
# =====================================================
class ViolenceDetector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.IMG_SIZE = (64, 64)
        self.hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

    def extract_features(self, prev_frame, curr_frame, prev_mag=None, prev_flow=None):
        if prev_frame is None or curr_frame is None:
            return None, None, None

        prev_gray = cv2.cvtColor(cv2.resize(prev_frame, self.IMG_SIZE), cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(cv2.resize(curr_frame, self.IMG_SIZE), cv2.COLOR_BGR2GRAY)

        hog_feat = self.hog.compute(curr_gray)
        if hog_feat is None: return None, None, None
        hog_feat = hog_feat.flatten()

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        valid = mag > 1.2

        mean_mag = acceleration = accel_jitter = std_mag = max_mag = motion_area = angle_std = symmetry = 0.0
        flow_consistency = directional_stability = rhythm_score = edge_density = 0.0
        
        if np.sum(valid) > 10:
            vmag = mag[valid]
            vang = ang[valid]
            mean_mag = np.mean(vmag)
            std_mag = np.std(vmag)
            max_mag = np.max(vmag)
            motion_area = np.sum(valid) / mag.size
            angle_std = np.std(vang)

            if prev_mag is not None:
                accel = mag - prev_mag
                vacc = accel[valid]
                acceleration = np.mean(np.abs(vacc))
                accel_jitter = np.std(vacc)

            h, w = mag.shape
            left = np.sum(mag[:, :w // 2])
            right = np.sum(mag[:, w // 2:])
            symmetry = abs(left - right) / (left + right + 1e-6)
            
            if prev_flow is not None:
                flow_diff = np.abs(flow - prev_flow)
                flow_consistency = np.mean(flow_diff[valid]) if np.sum(valid) > 10 else 0.0
            
            angle_bins = np.histogram(vang, bins=8, range=(0, 2*np.pi))[0]
            total = np.sum(angle_bins)
            if total > 0:
                probs = angle_bins / total
                angle_entropy = -np.sum(probs * np.log(probs + 1e-10))
                directional_stability = 1.0 / (angle_entropy + 1e-5)
            
            if len(vmag) > 16:
                fft = np.fft.fft(vmag[:256] if len(vmag) > 256 else vmag)
                power = np.abs(fft[:len(fft)//2])
                rhythm_score = np.max(power[1:]) / (np.mean(power) + 1e-5)
            
            edges = cv2.Canny(curr_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

        motion_feat = np.array([
            mean_mag, acceleration, accel_jitter, std_mag, 
            max_mag, motion_area, angle_std, symmetry,
            flow_consistency, directional_stability, rhythm_score, edge_density
        ])
        
        return np.hstack([hog_feat, motion_feat]), mag, flow

    def predict(self, feat):
        feat = feat.reshape(1, -1)
        return self.model.predict_proba(feat)[0][1]
    
    def calculate_final_prob(self, feat, raw_prob, is_realtime=False):
        """
        🔥 DUAL MODE:
        - is_realtime=False (Folder): Giữ nguyên logic gốc (accuracy cao)
        - is_realtime=True (Webcam): Thêm filters nghiêm ngặt
        """
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
        
        # ===============================================
        # WEBCAM MODE: EXTRA STRICT FILTERING
        # ===============================================
        if is_realtime:
            # Filter 1: Vẫy tay đơn giản
            if (symmetry > 0.4 and motion_area < 0.3 and mean_mag < 4.0):
                return raw_prob * 0.05
            
            # Filter 2: Lắc đầu
            if (rhythm_score > 2.0 and accel_jitter < 2.0 and motion_area < 0.25):
                return raw_prob * 0.08
            
            # Filter 3: Single motion zone
            if (motion_area < 0.35 and mean_mag < 5.0):
                return raw_prob * 0.15
            
            # Filter 4: Require multi-zone chaos
            if not (motion_area > 0.4 and accel_jitter > 2.5 and edge_density > 0.12):
                if raw_prob < 0.75:
                    return raw_prob * 0.25
        
        # ===============================================
        # SHARED LOGIC (Folder & Webcam)
        # ===============================================
        
        # TIER 1: STATIC DETECTION
        if mean_mag < 0.5:
            final_prob = raw_prob * (0.3 if raw_prob > 0.90 else 0.1)
        
        elif mean_mag < 0.8:
            if raw_prob > 0.85:
                final_prob = raw_prob * 0.5
            elif raw_prob > 0.7:
                final_prob = raw_prob * 0.3
            else:
                final_prob = raw_prob * 0.15
        
        # TIER 2: VIOLENCE PATTERNS
        elif (accel_jitter > 4.0 and std_mag > 3.0):
            final_prob = min(raw_prob * 1.6, 1.0)
        
        elif (accel_jitter > 3.5 and mean_mag > 2.0):
            final_prob = min(raw_prob * 1.5, 1.0)
        
        elif (accel_jitter > 3.0 and angle_std > 0.9):
            final_prob = min(raw_prob * 1.4, 1.0)
        
        elif (mean_mag > 2.0 and mean_mag < 4.5 and 
              accel_jitter > 2.5 and angle_std > 0.8):
            final_prob = min(raw_prob * 1.3, 1.0)
        
        elif (edge_density > 0.15 and accel_jitter > 2.8 and raw_prob > 0.70):
            final_prob = min(raw_prob * 1.35, 1.0)
        
        elif (flow_consistency > 3.0 and accel_jitter > 2.5):
            final_prob = min(raw_prob * 1.25, 1.0)
        
        # TIER 3: CLEAN SPORTS (STRONG PENALTY)
        elif (mean_mag > 4.0 and rhythm_score > 2.5 and 
              directional_stability > 0.8 and accel_jitter < 2.0):
            final_prob = raw_prob * 0.02
        
        elif (mean_mag > 5.0 and directional_stability > 1.0 and 
              flow_consistency < 1.5 and raw_prob < 0.65):
            final_prob = raw_prob * 0.03
        
        elif (mean_mag > 6.5 and angle_std < 0.55 and 
              accel_jitter < 1.8 and raw_prob < 0.60):
            final_prob = raw_prob * 0.03
        
        elif (mean_mag > 6.0 and symmetry < 0.3 and 
              rhythm_score > 2.0 and raw_prob < 0.65):
            final_prob = raw_prob * 0.04
        
        elif (mean_mag > 5.0 and angle_std < 0.60 and 
              accel_jitter < 2.0 and raw_prob < 0.70):
            final_prob = raw_prob * 0.08
        
        elif (mean_mag > 4.5 and std_mag < 1.5 and 
              accel_jitter < 1.5 and flow_consistency < 2.0):
            final_prob = raw_prob * 0.10
        
        # TIER 4: CONTACT SPORTS
        elif mean_mag > 4.5 and accel_jitter < 3.0 and raw_prob < 0.65:
            final_prob = raw_prob * 0.30
        
        elif mean_mag > 3.5 and accel_jitter < 2.5 and raw_prob < 0.70:
            final_prob = raw_prob * 0.50
        
        elif mean_mag > 2.5 and accel_jitter < 2.0:
            final_prob = raw_prob * 0.65
        
        elif mean_mag > 1.5 and mean_mag < 2.5:
            final_prob = raw_prob * 0.75
        
        else:
            final_prob = raw_prob
        
        return final_prob

# =====================================================
# GUI APP
# =====================================================
class App:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Violence Detection - Dual Mode (Folder Optimized)")
        self.root.geometry("1200x800")
        self.detector = ViolenceDetector(model_path)
        
        self.video_queue = []
        self.is_running = False
        self.cap = None
        self.mode = "file"
        self.tk_images = []
        
        self.last_alert_time = 0
        self.alert_cooldown = 2.0
        
        self.build_ui()
        self.reset_tracking_variables()

    def build_ui(self):
        main_container = tk.Frame(self.root, padx=10, pady=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(main_container, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH)

        right_panel = tk.LabelFrame(main_container, text="🎯 Evidence (High Confidence)")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        tk.Label(left_panel, text="INPUT SOURCE:", font=("Arial", 9, "bold")).pack(anchor="w")
        self.path_var = tk.StringVar()
        tk.Entry(left_panel, textvariable=self.path_var, width=45).pack(pady=5)
        
        btn_frame = tk.Frame(left_panel)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text="File", command=self.browse_file).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(btn_frame, text="Folder", command=self.browse_folder).pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.btn_start = tk.Button(left_panel, text="🚀 START", bg="#2ecc71", fg="white", command=self.start_process)
        self.btn_start.pack(pady=5, fill=tk.X)

        self.btn_camera = tk.Button(left_panel, text="📷 CAMERA (Strict Mode)", bg="#3498db", fg="white", command=self.start_camera)
        self.btn_camera.pack(pady=5, fill=tk.X)

        self.btn_stop = tk.Button(left_panel, text="⏹ STOP", bg="#e74c3c", fg="white", command=self.stop_process, state=tk.DISABLED)
        self.btn_stop.pack(pady=5, fill=tk.X)

        self.lbl_status = tk.Label(left_panel, text="Status: Idle", fg="blue")
        self.lbl_status.pack(anchor="w", pady=5)

        self.progress = ttk.Progressbar(left_panel, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        self.log = tk.Text(left_panel, height=20, width=45, font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_panel)
        self.scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

    def reset_tracking_variables(self):
        self.total_frames = 0
        self.processed = 0
        self.prev_frame = self.prev_mag = self.prev_flow = None
        self.smoothed_score = 0
        self.decay = 0.85
        self.max_score = 0
        self.top_frames = []
        
        self.static_frames = 0
        self.frames_with_motion = 0
        self.frames_with_strong_motion = 0
        self.frames_with_weak_motion = 0
        self.high_prob_frames = 0
        self.high_raw_prob_frames = 0
        self.chaos_frames = 0
        self.clean_sports_frames = 0
        self.consecutive_frames = 0
        self.max_consecutive = 0
        
        self.prob_history = []

    def browse_file(self):
        f = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov")])
        if f: self.path_var.set(f); self.mode = "file"

    def browse_folder(self):
        d = filedialog.askdirectory()
        if d: self.path_var.set(d); self.mode = "folder"

    def start_process(self):
        path = self.path_var.get()
        if not os.path.exists(path): 
            messagebox.showwarning("Error", "Path not found!")
            return
        
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.tk_images = []
        self.log.delete(1.0, tk.END)
        
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        
        if os.path.isdir(path):
            self.video_queue = [os.path.join(path, f) for f in os.listdir(path) 
                               if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            self.log.insert(tk.END, f"📁 Found {len(self.video_queue)} videos\n")
        else:
            self.video_queue = [path]
        
        self.next_video()

    def start_camera(self):
        self.is_running = True
        self.mode = "camera"
        self.btn_start.config(state=tk.DISABLED)
        self.btn_camera.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.cap = cv2.VideoCapture(0)
        self.reset_tracking_variables()
        self.log.insert(tk.END, "📷 Camera started (STRICT MODE)\n")
        self.log.insert(tk.END, "⚠️  Filters: Hand wave, head shake, single-zone motion\n\n")
        self.process_frame()

    def stop_process(self):
        self.is_running = False
        self.lbl_status.config(text="Status: Stopping...")

    def next_video(self):
        if not self.video_queue or not self.is_running:
            self.finish_all()
            return
        
        self.current_video_path = self.video_queue.pop(0)
        video_name = os.path.basename(self.current_video_path)
        self.lbl_status.config(text=f"Processing: {video_name}")
        self.log.insert(tk.END, f"\n▶ {video_name}\n")
        
        self.cap = cv2.VideoCapture(self.current_video_path)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress["maximum"] = total if total > 0 else 100
        self.reset_tracking_variables()
        self.process_frame()

    def process_frame(self):
        if not self.is_running or self.cap is None:
            self.finish_video()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.finish_video()
            return

        is_realtime = (self.mode == "camera")

        if self.prev_frame is not None:
            feat, curr_mag, curr_flow = self.detector.extract_features(
                self.prev_frame, frame, self.prev_mag, self.prev_flow
            )
            
            if feat is not None:
                raw_prob = self.detector.predict(feat)
                final_prob = self.detector.calculate_final_prob(feat, raw_prob, is_realtime)
                
                motion_stats = feat[-12:]
                mean_mag = motion_stats[0]
                accel_jitter = motion_stats[2]
                angle_std = motion_stats[6]
                
                if mean_mag < 0.8:
                    self.static_frames += 1
                elif mean_mag < 2.0:
                    self.frames_with_weak_motion += 1
                    self.frames_with_motion += 1
                elif mean_mag < 4.0:
                    self.frames_with_motion += 1
                else:
                    self.frames_with_strong_motion += 1
                    self.frames_with_motion += 1
                
                if accel_jitter > 3.0:
                    self.chaos_frames += 1
                
                if mean_mag > 5.5 and angle_std < 0.6 and accel_jitter < 1.5:
                    self.clean_sports_frames += 1
                
                if raw_prob > 0.65:
                    self.high_raw_prob_frames += 1
                
                if final_prob > 0.65:
                    self.high_prob_frames += 1
                
                if self.smoothed_score == 0:
                    self.smoothed_score = final_prob
                else:
                    self.smoothed_score = self.decay * self.smoothed_score + (1 - self.decay) * final_prob
                
                self.max_score = max(self.max_score, final_prob)
                
                threshold = 0.70 if is_realtime else 0.55
                if self.smoothed_score > threshold and mean_mag > 0.8:
                    self.consecutive_frames += 1
                else:
                    self.consecutive_frames = 0
                self.max_consecutive = max(self.max_consecutive, self.consecutive_frames)
                
                if is_realtime:
                    self.prob_history.append(final_prob)
                    if len(self.prob_history) > 30:
                        self.prob_history.pop(0)
                
                if final_prob > 0.65:
                    self.top_frames.append((final_prob, raw_prob, frame.copy(), self.processed))
                    self.top_frames.sort(key=lambda x: x[0], reverse=True)
                    self.top_frames = self.top_frames[:60]
                
                self.total_frames += 1
            
            self.prev_mag, self.prev_flow = curr_mag, curr_flow

        self.prev_frame = frame
        self.processed += 1
        self.progress["value"] = self.processed
        
        if self.mode == "camera":
            display = frame.copy()
            
            current_time = time.time()
            show_alert = False
            
            if len(self.prob_history) >= 15:
                recent_avg = np.mean(self.prob_history[-15:])
                recent_max = np.max(self.prob_history[-15:])
                
                if (self.consecutive_frames >= 8 and 
                    recent_avg > 0.75 and 
                    recent_max > 0.85 and
                    self.max_consecutive >= 8):
                    
                    if current_time - self.last_alert_time > self.alert_cooldown:
                        show_alert = True
                        self.last_alert_time = current_time
            
            if show_alert:
                cv2.rectangle(display, (0, 0), (display.shape[1], display.shape[0]), (0, 0, 255), 15)
                cv2.putText(display, "!!! VIOLENCE DETECTED !!!", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            
            status_text = f"Score: {self.smoothed_score:.1%} | Consec: {self.consecutive_frames}"
            color = (0, 0, 255) if self.smoothed_score > 0.70 else (0, 255, 0)
            cv2.putText(display, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("Live Feed", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.stop_process()

        self.root.after(1, self.process_frame)

    def finish_video(self):
        if self.cap: self.cap.release()
        if self.mode == "camera": cv2.destroyAllWindows()
        
        if self.total_frames == 0:
            if self.mode != "camera": 
                self.next_video()
            return
        
        # ===============================================
        # FOLDER MODE: OPTIMIZED DECISION LOGIC
        # ===============================================
        static_ratio = self.static_frames / self.total_frames
        motion_ratio = self.frames_with_motion / self.total_frames
        strong_motion_ratio = self.frames_with_strong_motion / self.total_frames
        high_prob_ratio = self.high_prob_frames / self.total_frames
        high_raw_prob_ratio = self.high_raw_prob_frames / self.total_frames
        chaos_ratio = self.chaos_frames / self.total_frames
        clean_sports_ratio = self.clean_sports_frames / self.total_frames
        
        is_viol = False
        decision_reason = ""
        
        # Branch 1: STATIC VIDEO (Giảm threshold từ 0.70 xuống 0.65)
        if static_ratio > 0.65:
            # Nới lỏng: Chỉ cần max_score > 0.85 hoặc high_raw_prob > 0.55
            is_viol = (self.max_score > 0.85 and high_raw_prob_ratio > 0.50) or \
                      (self.max_score > 0.90 and high_raw_prob_ratio > 0.40)
            decision_reason = "STATIC"
        
        # Branch 2: MOSTLY STATIC (Giảm từ 0.50 xuống 0.45)
        elif static_ratio > 0.45:
            # Nới lỏng: max_consecutive >= 5 (từ 10), smoothed > 0.70 (từ 0.80)
            is_viol = (self.max_consecutive >= 5 and self.smoothed_score > 0.70) or \
                      (self.max_score > 0.90 and high_prob_ratio > 0.25)
            decision_reason = "MOSTLY_STATIC"
        
        # Branch 3: CHAOS-DOMINATED (Giữ nguyên 0.25)
        elif chaos_ratio > 0.25:
            is_viol = (self.max_consecutive >= 4) or \
                      (self.smoothed_score > 0.60) or \
                      (self.max_score > 0.75) or \
                      (high_prob_ratio > 0.20 and self.max_score > 0.65)
            decision_reason = "CHAOS"
        
        # Branch 4: CLEAN SPORTS (Giữ nguyên - cần strict)
        elif clean_sports_ratio > 0.40 or (strong_motion_ratio > 0.50 and chaos_ratio < 0.15):
            is_viol = (self.max_consecutive >= 10 and self.smoothed_score > 0.90) or \
                      (self.smoothed_score > 0.90 and self.max_score > 0.98) or \
                      (chaos_ratio > 0.35 and self.max_score > 0.95)
            decision_reason = "CLEAN_SPORTS"
        
        # Branch 5: HIGH MOTION / CONTACT SPORTS (Nới lỏng)
        elif strong_motion_ratio > 0.35:
            # Giảm threshold: max_consecutive >= 4 (từ 6), smoothed > 0.70 (từ 0.75)
            is_viol = (self.max_consecutive >= 4) or \
                      (self.smoothed_score > 0.70) or \
                      (self.max_score > 0.85) or \
                      (chaos_ratio > 0.15 and self.max_score > 0.75)
            decision_reason = "CONTACT_SPORTS"
        
        # Branch 6: NORMAL MOTION (Nới lỏng)
        elif motion_ratio > 0.20:
            # Spike safeguard vẫn giữ
            if self.max_score > 0.88 and self.smoothed_score < 0.55 and self.max_consecutive < 3:
                is_viol = False
            else:
                # Giảm: max_consecutive >= 4 (từ 6), smoothed > 0.70 (từ 0.75)
                is_viol = (self.max_consecutive >= 4) or \
                          (self.smoothed_score > 0.70) or \
                          (self.max_score > 0.88) or \
                          (high_prob_ratio > 0.30 and self.max_score > 0.80)
            decision_reason = "NORMAL"
        
        # Branch 7: LOW MOTION (Nới lỏng)
        elif motion_ratio > 0.10:
            # Giảm: max_consecutive >= 3 (từ 5), smoothed > 0.65 (từ 0.70)
            is_viol = (self.max_consecutive >= 3) or \
                      (self.smoothed_score > 0.65) or \
                      (self.max_score > 0.80) or \
                      (high_raw_prob_ratio > 0.35 and self.max_score > 0.70)
            decision_reason = "LOW_MOTION"
        
        # Branch 8: MINIMAL MOTION (Nới lỏng)
        else:
            # Giảm: high_raw_prob > 0.45 (từ 0.50), max_score > 0.75 (từ 0.80)
            is_viol = (high_raw_prob_ratio > 0.45 and self.max_score > 0.75)
            decision_reason = "MINIMAL"
        
        # Logging
        video_name = os.path.basename(getattr(self, 'current_video_path', 'Camera'))
        color_tag = "red" if is_viol else "green"
        
        self.log.insert(tk.END, f"{'🚨 VIOLENCE' if is_viol else '✅ SAFE'} ({decision_reason})\n", color_tag)
        self.log.insert(tk.END, f"  Max: {self.max_score:.1%} | Smooth: {self.smoothed_score:.1%} | Consec: {self.max_consecutive}\n")
        self.log.insert(tk.END, f"  Motion: {motion_ratio:.0%} | Chaos: {chaos_ratio:.0%} | Sports: {clean_sports_ratio:.0%}\n")
        self.log.insert(tk.END, "-"*40 + "\n")
        
        self.log.tag_config("red", foreground="red", font=("Arial", 9, "bold"))
        self.log.tag_config("green", foreground="green", font=("Arial", 9, "bold"))
        self.log.see(tk.END)
        
        # Display evidence only if violence
        if is_viol and self.top_frames:
            self.display_evidence(video_name)
        
        if self.mode != "camera" and self.is_running:
            self.root.after(100, self.next_video)
        else:
            self.finish_all()

    def display_evidence(self, video_title):
        if not self.top_frames: return
        
        # Filter: final_prob > 90% first, then 80%, then 65%
        candidates = [f for f in self.top_frames if f[0] > 0.90]
        threshold = 90
        if not candidates:
            candidates = [f for f in self.top_frames if f[0] > 0.80]
            threshold = 80
        if not candidates:
            candidates = [f for f in self.top_frames if f[0] > 0.65]
            threshold = 65
        
        # Diversity filter
        selected = []
        min_dist = 30
        candidates.sort(key=lambda x: x[3])
        for cand in candidates:
            if not selected or abs(cand[3] - selected[-1][3]) > min_dist:
                selected.append(cand)
        
        final_list = selected[:5]
        
        if not final_list: return
        
        color = "#c0392b" if threshold == 90 else "#d35400" if threshold == 80 else "#e67e22"
        
        container = tk.Frame(self.scrollable_frame, pady=10, bg="#f0f0f0", 
                           highlightbackground="gray", highlightthickness=1)
        container.pack(fill=tk.X, pady=5, padx=5)

        header = tk.Label(container, text=f"📂 {video_title} (>{threshold}%)", 
                         bg=color, fg="white", font=("Arial", 9, "bold"))
        header.pack(fill=tk.X)

        images_frame = tk.Frame(container, bg="#f0f0f0")
        images_frame.pack()
        
        for final_prob, raw_prob, img, idx in final_list:
            item = tk.Frame(images_frame, bg="#f0f0f0", padx=5)
            item.pack(side=tk.LEFT)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((180, 135))
            tk_img = ImageTk.PhotoImage(img_pil)
            self.tk_images.append(tk_img)

            tk.Label(item, image=tk_img, bd=2, relief="ridge").pack()
            tk.Label(item, text=f"Final: {final_prob:.1%}", fg="red", 
                    font=("Arial", 8, "bold"), bg="#f0f0f0").pack()
            tk.Label(item, text=f"Raw: {raw_prob:.1%}", fg="gray", 
                    font=("Arial", 7), bg="#f0f0f0").pack()

        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def finish_all(self):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_camera.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_status.config(text="Status: Finished")

if __name__ == "__main__":
    root = tk.Tk()
    MODEL_PATH = r"C:\Users\admin\Desktop\Xla\violence_ensemble_model_final.joblib"
    if os.path.exists(MODEL_PATH):
        App(root, MODEL_PATH)
        root.mainloop()
    else:
        messagebox.showerror("Error", f"Model not found at {MODEL_PATH}")