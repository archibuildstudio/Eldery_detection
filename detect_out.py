import cv2
from ultralytics import YOLO

# ------------------------------------------------------------
# 1.  Konfigurasi
# ------------------------------------------------------------
CLASSES_TXT  = 'classes.txt'            # daftar label YOLO (satu per baris)
MODEL_PATH   = 'yolo11n.pt'               # model YOLO
INPUT_VIDEO  = 'maul.mp4'                 # video sumber

def load_class_names(path: str):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]



def point_inside(px, py, x1, y1, x2, y2):
    return x1 < px < x2 and y1 < py < y2


# ------------------------------------------------------------
# 2.  Inisialisasi
# ------------------------------------------------------------
class_names = load_class_names(CLASSES_TXT)
model       = YOLO(MODEL_PATH)
cap         = cv2.VideoCapture(INPUT_VIDEO)

w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25


# ---- koordinat Bed‑1 (ubah sesuai posisi ranjang)
bed_tl, bed_br = (200, 100), (1000, 600)

# ------------------------------------------------------------
# 3.  Loop deteksi
# ------------------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    bed_has_person = False

    # gambar region
    cv2.rectangle(frame, bed_tl, bed_br, (0, 255, 0), 2)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf           = float(box.conf[0])
            cls_idx        = int(box.cls[0])
            cls_name       = class_names[cls_idx]

            # hanya cek kelas 'person'
            if cls_name == 'person' and conf > 0.25:
                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                if point_inside(cx, cy, *bed_tl, *bed_br):
                    bed_has_person = True

            # (opsional) tampilkan semua bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(frame, cls_name, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # status Bed‑1
    if bed_has_person:
        msg, color = "Bed‑1 NOT empty", (0, 255, 0)
    else:
        msg, color = "Bed‑1 EMPTY", (0, 0, 255)
    cv2.putText(frame, msg, (bed_tl[0], bed_tl[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.rectangle(frame, bed_tl, bed_br, color, 2)

    # simpan & tampil
    cv2.imshow('Bed‑1 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------------------------------------------------
# 4.  Cleanup
# ------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()

