import sys
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import joblib
import torch
import torch.nn.functional as F
import cv2

# --- FIX PATH ISSUES FOR EXECUTABLE ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Ensure the app can find your local modules during development
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from src.data.features import hog_features
    from src.models.cnn_lenet import LeNetMNIST
    from src.models.mlp import MLP
except ImportError:
    # If imports fail inside the EXE, ensure they are handled or bundled
    pass 

# -----------------------
# Global Configuration
# -----------------------
# Switch paths based on whether we are running as an EXE or a Script
if getattr(sys, 'frozen', False):
    # Inside EXE: Models are mapped to 'models' root via --add-data
    MODEL_DIR = resource_path("models")
else:
    # During Development: Models are in outputs/models
    MODEL_DIR = os.path.join("outputs", "models")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# COLORS (Dark Green Theme)
BG_COLOR = "#e8f5e9"
CARD_COLOR = "#ffffff"
ACCENT_COLOR = "#1b5e20"
BTN_COLOR = "#2e7d32"
TEXT_COLOR = "#1b5e20"
SUCCESS_COLOR = "#43a047"

# -----------------------
# Model Loading Logic
# -----------------------
models = {}

def load_models():
    """Loads models safely using the resource_path system."""
    # Load SVM
    try:
        svm_path = os.path.join(MODEL_DIR, "svm_hog.pkl")
        models['SVM (Traditional)'] = joblib.load(svm_path)
    except Exception as e:
        print(f"SVM failed to load: {e}")

    # Load CNN
    try:
        cnn_path = os.path.join(MODEL_DIR, "cnn_lenet.pt")
        cnn = LeNetMNIST().to(DEVICE)
        cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
        cnn.eval()
        models['CNN (Deep Learning)'] = cnn
    except Exception as e:
        print(f"CNN failed to load: {e}")

    # Load MLP
    try:
        mlp_path = os.path.join(MODEL_DIR, "mlp.pt")
        mlp = MLP().to(DEVICE)
        mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
        mlp.eval()
        models['MLP (Neural Net)'] = mlp
    except Exception as e:
        print(f"MLP failed to load: {e}")

# -----------------------
# Custom Widgets
# -----------------------
class ModernButton(tk.Button):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.config(font=("Segoe UI", 10, "bold"), bg=BTN_COLOR, fg="white", 
                    activebackground=ACCENT_COLOR, activeforeground="white", 
                    bd=0, padx=15, pady=8, cursor="hand2")

# -----------------------
# Main GUI
# -----------------------
class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.geometry("800x700")
        self.root.configure(bg=BG_COLOR)

        self.image_path = None
        self.processed_img = None

        header_frame = tk.Frame(root, bg=ACCENT_COLOR, height=80)
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="Handwritten Digit Recognizer", font=("Segoe UI", 18, "bold"), 
                 bg=ACCENT_COLOR, fg="white").pack(pady=20)

        container = tk.Frame(root, bg=BG_COLOR)
        container.pack(fill="both", expand=True, padx=30, pady=20)

        left_col = tk.Frame(container, bg=BG_COLOR)
        left_col.pack(side="left", fill="both", expand=True, padx=10)

        # Step 1: Input
        self.create_card(left_col, "Step 1: Input")
        ModernButton(self.card_frame, text="📂 Upload Image", command=self.load_image).pack(pady=10)
        
        # Step 2: Analysis
        self.create_card(left_col, "Step 2: Analysis")
        tk.Label(self.card_frame, text="Choose your algorithm:", bg=CARD_COLOR, font=("Segoe UI", 10)).pack(anchor="w")
        
        self.model_var = tk.StringVar()
        # Populating dropdown keys
        model_keys = list(models.keys())
        if model_keys: 
            self.model_var.set(model_keys[0])
        
        style = ttk.Style()
        style.theme_use('clam')
        self.combo = ttk.Combobox(self.card_frame, textvariable=self.model_var, 
                                  values=model_keys, state="readonly", font=("Segoe UI", 10))
        self.combo.pack(fill="x", pady=10)
        
        ModernButton(self.card_frame, text="🔮 Predict Digit", command=self.predict).pack(fill="x", pady=5)

        right_col = tk.Frame(container, bg=BG_COLOR)
        right_col.pack(side="right", fill="both", expand=True, padx=10)

        # Visualization
        self.create_card(right_col, "Visualization")
        vis_frame = tk.Frame(self.card_frame, bg=CARD_COLOR)
        vis_frame.pack(pady=10)

        self.panel_orig = tk.Label(vis_frame, bg="#eee", text="Original", width=18, height=9)
        self.panel_orig.grid(row=0, column=0, padx=5)
        
        self.panel_proc = tk.Label(vis_frame, bg="black", fg="white", text="Processed", width=18, height=9)
        self.panel_proc.grid(row=0, column=1, padx=5)

        tk.Label(self.card_frame, text="PREDICTION", font=("Segoe UI", 10, "bold"), fg="#aaa", bg=CARD_COLOR).pack(pady=(20, 0))
        self.lbl_result = tk.Label(self.card_frame, text="?", font=("Segoe UI", 48, "bold"), fg=ACCENT_COLOR, bg=CARD_COLOR)
        self.lbl_result.pack()
        
        self.lbl_conf = tk.Label(self.card_frame, text="Confidence: --%", font=("Segoe UI", 12), fg="#888", bg=CARD_COLOR)
        self.lbl_conf.pack(pady=(0, 20))

        footer_frame = tk.Frame(root, bg="#c8e6c9", height=50)
        footer_frame.pack(side="bottom", fill="x")
        tk.Button(footer_frame, text="📊 View Accuracy Charts", command=self.show_metrics, 
                  relief="flat", bg="#c8e6c9", fg=ACCENT_COLOR, font=("Segoe UI", 9, "bold")).pack(side="right", padx=20, pady=10)

    def create_card(self, parent, title):
        frame = tk.Frame(parent, bg=CARD_COLOR, bd=1, relief="solid")
        frame.pack(fill="x", pady=10, ipady=10)
        frame.config(highlightbackground="#a5d6a7", highlightthickness=1, relief="flat")
        tk.Label(frame, text=title, font=("Segoe UI", 11, "bold"), bg=CARD_COLOR, fg=ACCENT_COLOR).pack(anchor="w", padx=15, pady=5)
        tk.Frame(frame, bg="#e0e0e0", height=1).pack(fill="x", padx=10)
        self.card_frame = tk.Frame(frame, bg=CARD_COLOR)
        self.card_frame.pack(fill="both", expand=True, padx=15, pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        self.image_path = path
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
        self.processed_img = resized.astype(np.float32) / 255.0
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb).resize((130, 130))
        imgtk = ImageTk.PhotoImage(im_pil)
        self.panel_orig.config(image=imgtk, width=130, height=130, text="")
        self.panel_orig.image = imgtk
        im_proc_pil = Image.fromarray(resized).resize((130, 130))
        imgtk_proc = ImageTk.PhotoImage(im_proc_pil)
        self.panel_proc.config(image=imgtk_proc, width=130, height=130, text="")
        self.panel_proc.image = imgtk_proc
        self.lbl_result.config(text="?")
        self.lbl_conf.config(text="Ready to predict")

    def predict(self):
        if self.processed_img is None:
            messagebox.showwarning("Oops!", "Please upload an image first.")
            return
        name = self.model_var.get()
        if not name: return
        model = models[name]
        try:
            if "SVM" in name:
                features = hog_features(self.processed_img[np.newaxis, :, :])
                probs = model.predict_proba(features)[0]
                pred = np.argmax(probs)
                conf = probs[pred]
            else:
                t = torch.tensor(self.processed_img).unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs = F.softmax(model(t), dim=1).cpu().numpy()[0]
                    pred = np.argmax(probs)
                    conf = probs[pred]
            self.lbl_result.config(text=str(pred))
            self.lbl_conf.config(text=f"Confidence: {conf*100:.1f}%")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_metrics(self):
        messagebox.showinfo("Metrics", "Run 'src/eval/make_report_figures.py' to generate full charts!")

if __name__ == "__main__":
    load_models()
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
    root.mainloop()