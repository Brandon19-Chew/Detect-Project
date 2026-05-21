from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from datetime import datetime

model = YOLO("yolov8s.pt")

root = tk.Tk()
root.title("Human Detection System")
root.geometry("1100x750")
root.configure(bg="#1a1a2e")

DARK_BG    = "#1a1a2e"
PANEL_BG   = "#16213e"
ACCENT     = "#0f3460"
GREEN      = "#00d26a"
CARD_BG    = "#0f3460"
TEXT_WHITE = "#e0e0e0"
TEXT_DIM   = "#8888aa"
current_photo = None

# ── Header ────────────────────────────────────────────────────────────────────
header = tk.Frame(root, bg=DARK_BG)
header.pack(fill="x", padx=20, pady=(16, 0))
tk.Label(header, text="Human Detection System", font=("Helvetica", 22, "bold"),
         bg=DARK_BG, fg=TEXT_WHITE).pack(side="left")
tk.Label(header, text="Powered by YOLOv8s", font=("Helvetica", 11),
         bg=DARK_BG, fg=TEXT_DIM).pack(side="left", padx=(12, 0), pady=(6, 0))
tk.Button(header, text="Upload Image", font=("Helvetica", 12, "bold"),
          bg=GREEN, fg="#000000", padx=18, pady=8, relief="flat",
          cursor="hand2", command=lambda: upload_image()).pack(side="right")
tk.Frame(root, height=1, bg=ACCENT).pack(fill="x", padx=20, pady=12)

# ── Layout ────────────────────────────────────────────────────────────────────
main = tk.Frame(root, bg=DARK_BG)
main.pack(fill="both", expand=True, padx=20, pady=(0, 16))

left = tk.Frame(main, bg=PANEL_BG)
left.pack(side="left", fill="both", expand=True, padx=(0, 12))
img_label_title = tk.Label(left, text="No image loaded", font=("Helvetica", 11),
                            bg=PANEL_BG, fg=TEXT_DIM)
img_label_title.pack(anchor="w", padx=12, pady=(10, 4))
img_label = tk.Label(left, bg="#0a0a1a", width=60, height=28)
img_label.pack(padx=12, pady=(0, 12), fill="both", expand=True)

right = tk.Frame(main, bg=DARK_BG, width=300)
right.pack(side="right", fill="y")
right.pack_propagate(False)
tk.Label(right, text="Detection Results", font=("Helvetica", 14, "bold"),
         bg=DARK_BG, fg=TEXT_WHITE).pack(anchor="w", pady=(0, 8))

summary_frame = tk.Frame(right, bg=DARK_BG)
summary_frame.pack(fill="x", pady=(0, 12))

def make_stat_card(parent, label, var):
    card = tk.Frame(parent, bg=CARD_BG, padx=14, pady=10)
    card.pack(side="left", expand=True, fill="x", padx=(0, 8))
    tk.Label(card, text=label, font=("Helvetica", 9),  bg=CARD_BG, fg=TEXT_DIM).pack(anchor="w")
    tk.Label(card, textvariable=var, font=("Helvetica", 22, "bold"), bg=CARD_BG, fg=GREEN).pack(anchor="w")

total_var    = tk.StringVar(value="—")
avg_conf_var = tk.StringVar(value="—")
make_stat_card(summary_frame, "Total Humans",   total_var)
make_stat_card(summary_frame, "Avg Confidence", avg_conf_var)

tk.Label(right, text="Per-Person Details", font=("Helvetica", 12, "bold"),
         bg=DARK_BG, fg=TEXT_WHITE).pack(anchor="w", pady=(4, 6))
table_frame = tk.Frame(right, bg=DARK_BG)
table_frame.pack(fill="both", expand=True)
cols = ("ID", "Confidence", "Width", "Height", "Area", "Position")
tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=14)
style = ttk.Style()
style.theme_use("clam")
style.configure("Treeview", background=PANEL_BG, foreground=TEXT_WHITE,
                fieldbackground=PANEL_BG, rowheight=26, font=("Helvetica", 10))
style.configure("Treeview.Heading", background=ACCENT, foreground=TEXT_WHITE,
                font=("Helvetica", 10, "bold"), relief="flat")
style.map("Treeview", background=[("selected", ACCENT)])
for col, w in zip(cols, [30, 80, 50, 55, 65, 85]):
    tree.heading(col, text=col)
    tree.column(col, width=w, anchor="center")
scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
tree.configure(yscrollcommand=scrollbar.set)
tree.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

meta_frame = tk.Frame(right, bg=DARK_BG)
meta_frame.pack(fill="x", pady=(10, 0))
file_var = tk.StringVar(value="")
time_var = tk.StringVar(value="")
tk.Label(meta_frame, textvariable=file_var, font=("Helvetica", 9), bg=DARK_BG, fg=TEXT_DIM).pack(anchor="w")
tk.Label(meta_frame, textvariable=time_var, font=("Helvetica", 9), bg=DARK_BG, fg=TEXT_DIM).pack(anchor="w")

# ── Detection core ────────────────────────────────────────────────────────────
def get_position(cx, cy, img_w, img_h):
    v = "Top"  if cy < img_h/3 else ("Bottom" if cy > 2*img_h/3 else "Middle")
    h = "Left" if cx < img_w/3 else ("Right"  if cx > 2*img_w/3 else "Center")
    return f"{v}-{h}"

def iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def non_max_suppression(detections, iou_threshold=0.45):
    """
    Apply NMS to remove duplicate detections.
    Using standard threshold of 0.45 (not 0.30 which is too aggressive).
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence descending
    detections = sorted(detections, key=lambda x: x[0], reverse=True)
    
    keep = []
    while len(detections) > 0:
        # Take the highest confidence detection
        best = detections.pop(0)
        keep.append(best)
        
        # Remove detections that overlap too much with the best one
        filtered = []
        for det in detections:
            if iou(best[1], det[1]) < iou_threshold:
                filtered.append(det)
        detections = filtered
    
    return keep

def detect(img):
    """
    Robust detection strategy:
    
    For ALL images:
    1. Run YOLO at native resolution with standard confidence threshold
    2. For small images (longest side < 400px), also run an upscaled version
    3. Merge results and apply NMS
    
    This approach is more reliable than strip-based detection.
    """
    h, w = img.shape[:2]
    all_detections = []
    
    # ── Always run at native resolution ─────────────────────────────
    # Standard YOLO detection
    results = model(img, conf=0.25, iou=0.45, imgsz=640, verbose=False)
    
    for r in results:
        for box in r.boxes:
            # Only keep person class (class 0 in COCO)
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                all_detections.append((confidence, [x1, y1, x2, y2]))
    
    # ── For small images, also run upscaled version ─────────────────
    if max(w, h) < 400:
        # Upscale small images to help YOLO detect small people
        scale_factor = min(640 / max(w, h), 3.0)  # Cap scaling at 3x
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Ensure dimensions are multiples of 32 (YOLO requirement)
        new_w = (new_w // 32) * 32
        new_h = (new_h // 32) * 32
        
        img_upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Run detection on upscaled image with slightly lower confidence threshold
        results_upscaled = model(img_upscaled, conf=0.20, iou=0.45, imgsz=640, verbose=False)
        
        for r in results_upscaled:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    
                    # Scale coordinates back to original image size
                    x1 = x1 / scale_factor
                    y1 = y1 / scale_factor
                    x2 = x2 / scale_factor
                    y2 = y2 / scale_factor
                    
                    all_detections.append((confidence, [x1, y1, x2, y2]))
    
    # ── Apply NMS to merge overlapping detections ────────────────────
    # Use standard NMS threshold (0.45) instead of aggressive 0.30
    final_detections = non_max_suppression(all_detections, iou_threshold=0.45)
    
    return final_detections

# ── Upload handler ────────────────────────────────────────────────────────────
def upload_image():
    global current_photo
    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                   ("All Files", "*.*")]
    )
    if not path:
        return

    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", "Could not load image.")
        return

    # Create a copy for drawing boxes
    display_img = img.copy()
    img_h, img_w = img.shape[:2]
    
    t0 = datetime.now()
    detections = detect(img)
    elapsed = (datetime.now() - t0).total_seconds()

    # Clear previous results
    for row in tree.get_children():
        tree.delete(row)

    human_count = 0
    conf_scores = []

    for conf, (x1, y1, x2, y2) in detections:
        human_count += 1
        
        # Clip coordinates to image boundaries
        x1 = max(0, min(img_w, int(round(x1))))
        y1 = max(0, min(img_h, int(round(y1))))
        x2 = max(0, min(img_w, int(round(x2))))
        y2 = max(0, min(img_h, int(round(y2))))
        
        conf_scores.append(conf)

        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        pos = get_position(cx, cy, img_w, img_h)

        # Draw bounding box and label
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 210, 106), 2)
        label = f"#{human_count}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(display_img, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 210, 106), -1)
        cv2.putText(display_img, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        tree.insert("", "end", values=(
            f"#{human_count}", f"{conf:.1%}",
            f"{bw}px", f"{bh}px", f"{area:,}", pos
        ))

    # Update statistics
    total_var.set(str(human_count))
    if conf_scores:
        avg_conf = sum(conf_scores) / len(conf_scores)
        avg_conf_var.set(f"{avg_conf:.0%}")
    else:
        avg_conf_var.set("—")
    
    file_var.set(f"File: {os.path.basename(path)}")
    time_var.set(f"{datetime.now().strftime('%H:%M:%S')}  |  {img_w}×{img_h}px  |  {elapsed:.1f}s")
    img_label_title.config(text=f"{os.path.basename(path)}  —  {img_w}×{img_h}")

    # Display image
    rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    panel_w = left.winfo_width() or 700
    panel_h = left.winfo_height() or 520
    pil.thumbnail((panel_w - 24, panel_h - 50), Image.LANCZOS)
    current_photo = ImageTk.PhotoImage(pil)
    img_label.config(image=current_photo)

root.mainloop()
