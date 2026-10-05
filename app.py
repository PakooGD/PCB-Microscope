import os, io, time, json, queue, threading
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import serial
from serial.tools import list_ports
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ------------- ENV -------------
GRBL_PORT    = os.environ.get("GRBL_PORT", "AUTO")
GRBL_BAUD    = int(os.environ.get("GRBL_BAUD", "115200"))
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAM_WIDTH    = int(os.environ.get("CAM_WIDTH", "1920"))
CAM_HEIGHT   = int(os.environ.get("CAM_HEIGHT", "1080"))
CAPTURES_DIR = Path(os.environ.get("CAPTURES_DIR", "captures")).resolve()
CAPTURES_DIR.mkdir(exist_ok=True)


# ======================= GRBL =======================
class GRBLController:
    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        self.q: "queue.Queue[str]" = queue.Queue()
        self.cmd_lock = threading.Lock()
        self.status = {"state": "Unknown", "x": 0.0, "y": 0.0, "z": 0.0}
        self._stop = False
        self._wake()
        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def _wake(self):
        self.ser.write(b"\r\n\r\n")
        time.sleep(1)
        self.ser.reset_input_buffer()

    def _read_loop(self):
        buf = ""
        while not self._stop:
            try:
                data = self.ser.read(512)
            except Exception:
                time.sleep(0.1); continue
            if not data:
                continue
            buf += data.decode(errors="ignore")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith("<"):
                    self._parse_status(line)
                else:
                    self.q.put(line)

    def _poll_loop(self):
        while not self._stop:
            try:
                self.ser.write(b"?")
            except Exception:
                pass
            time.sleep(0.25)

    def _parse_status(self, line: str):
        try:
            body = line[1:-1]
            parts = body.split("|")
            state = parts[0]
            mpos = None
            for p in parts:
                if p.startswith("MPos:"):
                    mpos = [float(v) for v in p[5:].split(",")]
            if mpos:
                self.status = {"state": state, "x": mpos[0], "y": mpos[1], "z": mpos[2]}
            else:
                self.status["state"] = state
        except Exception:
            pass

    def send(self, cmd: str, timeout: float = 60.0, wait_ok: bool = True) -> bool:
        with self.cmd_lock:
            self.ser.write((cmd + "\n").encode())
            if not wait_ok:
                return True
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    line = self.q.get(timeout=0.3)
                except queue.Empty:
                    continue
                if line == "ok":
                    return True
                if line.startswith("error") or line.startswith("ALARM"):
                    raise RuntimeError(line)
            raise TimeoutError(f"No OK for {cmd!r}")

    def home(self):
        self.send("$H", timeout=180)

    def unlock(self):
        self.send("$X", timeout=10)

    def move_abs(self, x: float, y: float, feed: float):
        self.send(f"G90 G0 X{x:.3f} Y{y:.3f} F{feed:.0f}", timeout=120)

    def move_rel(self, dx: float, dy: float, feed: float):
        self.send(f"G91 G0 X{dx:.3f} Y{dy:.3f} F{feed:.0f}", timeout=120)

    def wait_idle(self, timeout: float = 120.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.status.get("state") in ("Idle", "Check"):
                return
            time.sleep(0.1)
        raise TimeoutError("GRBL not idle")

    def close(self):
        self._stop = True
        try: self.ser.close()
        except Exception: pass


# ======================= Camera =======================
class Camera:
    def __init__(self, index: int = 0, w: int = 1920, h: int = 1080):
        self.cap = self._open(index)
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(5):
            self.cap.read()
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self._stop = False
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _open(index: int):
        if os.name == "nt":
            backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]
        else:
            backends = [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]
        for name, backend in backends:
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        print(f"[camera] opened index={index} via {name}")
                        return cap
                    cap.release()
            except Exception as e:
                print(f"[camera] backend {name} failed: {e}")
        return None

    def _loop(self):
        while not self._stop:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f
            else:
                time.sleep(0.05)

    def get(self) -> Optional[np.ndarray]:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        self._stop = True
        try: self.cap.release()
        except Exception: pass


# ======================= State =======================
STATE: Dict[str, Any] = {
    "grbl": None,
    "camera": None,
    "brightness": 0,
    "contrast": 0,
    "saturation": 0,
    "session_dir": None,
    "captures": [],
    "grid_cols": 0,
    "grid_rows": 0,
    "naming": "coords",   # "coords" | "seq"
    "seq": 0,             # счётчик для режима "seq"
}

SCAN = {"running": False, "progress": 0, "total": 0, "message": ""}
DEMO = {
    "running": False,
    "row": 0, "col": 0,          # текущая ячейка сетки
    "cycle": 0,                   # номер прохода (цикл)
    "tick": 0,                    # счётчик сработавших «затворов»
    "phase": "idle",              # idle | moving | settling | shooting
    "shot_at": 0.0,               # time.time() последней «съёмки»
    "message": "",
}

def apply_adjust(frame: np.ndarray, brightness: int, contrast: int, saturation: int) -> np.ndarray:
    if brightness == 0 and contrast == 0 and saturation == 0:
        return frame
    img = frame.astype(np.float32)
    if brightness != 0:
        img += brightness
    if contrast != 0:
        c = contrast * 1.28
        f = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c))
        img = f * (img - 128.0) + 128.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    if saturation != 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + saturation / 100.0), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img

def make_filename(row: int, col: int, tag: Optional[str] = None) -> str:
    """Возвращает имя файла в зависимости от выбранного режима именования."""
    if STATE.get("naming") == "seq":
        name = f"{STATE['seq']:04d}.png"
        STATE["seq"] += 1
        return name
    # режим "coords"
    if tag is None:
        return f"r{row:03d}_c{col:03d}.png"
    return f"{tag}.png"

def new_session() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = CAPTURES_DIR / f"session_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    STATE["session_dir"] = p
    STATE["captures"] = []
    STATE["grid_cols"] = 0
    STATE["grid_rows"] = 0
    STATE["seq"] = 0
    return p


# ======================= Scan =======================
def run_scan(rows: int, cols: int, step_x: float, step_y: float,
             feed: float, snake: bool, settle: float):
    grbl: Optional[GRBLController] = STATE["grbl"]
    cam: Optional[Camera] = STATE["camera"]
    if grbl is None:
        SCAN.update(running=False, message="GRBL not connected")
        return
    if cam is None:
        SCAN.update(running=False, message="Camera not connected")
        return

    SCAN.update(running=True, total=rows * cols, progress=0, message="Homing...")
    session = new_session()
    STATE["grid_rows"] = rows
    STATE["grid_cols"] = cols

    try:
        grbl.wait_idle()
        x0 = grbl.status.get("x", 0.0)
        y0 = grbl.status.get("y", 0.0)

        for r in range(rows):
            seq = list(range(cols))
            if snake and r % 2 == 1:
                seq.reverse()
            for c in seq:
                if not SCAN["running"]:
                    return
                x = x0 + c * step_x
                y = y0 + r * step_y
                grbl.move_abs(x, y, feed)
                try:
                    grbl.wait_idle(timeout=60)
                except Exception:
                    pass
                time.sleep(settle)

                frame = cam.get()
                if frame is None:
                    continue
                frame = apply_adjust(frame, STATE["brightness"],
                                     STATE["contrast"], STATE["saturation"])
                fname = make_filename(r, c)
                cv2.imwrite(str(session / fname), frame)
                STATE["captures"].append({
                    "row": r, "col": c, "file": fname, "x": x, "y": y
                })
                SCAN["progress"] += 1
                SCAN["message"] = f"{SCAN['progress']}/{SCAN['total']}"

        with open(session / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "rows": rows, "cols": cols,
                "step_x": step_x, "step_y": step_y,
                "brightness": STATE["brightness"],
                "contrast": STATE["contrast"],
                "saturation": STATE["saturation"],
                "captures": STATE["captures"],
            }, f, indent=2)
        SCAN["message"] = "Done"
    except Exception as e:
        SCAN["message"] = f"Error: {e}"
    finally:
        SCAN["running"] = False

def run_demo(rows: int, cols: int, step_x: float, step_y: float,
             feed: float, snake: bool, settle: float):
    grbl: Optional[GRBLController] = STATE["grbl"]
    if grbl is None:
        DEMO.update(running=False, phase="idle", message="GRBL not connected")
        return

    DEMO.update(running=True, row=0, col=0, cycle=0, tick=0,
                phase="idle", shot_at=0.0, message="Demo running")
    STATE["grid_rows"] = rows
    STATE["grid_cols"] = cols

    try:
        while DEMO["running"]:
            DEMO["cycle"] += 1
            # «домашняя» точка — берём текущую позицию как старт каждого прохода
            try:
                grbl.wait_idle(timeout=5)
            except Exception:
                pass
            x0 = grbl.status.get("x", 0.0)
            y0 = grbl.status.get("y", 0.0)

            for r in range(rows):
                if not DEMO["running"]:
                    return
                seq = list(range(cols))
                if snake and r % 2 == 1:
                    seq.reverse()
                for c in seq:
                    if not DEMO["running"]:
                        return

                    x = x0 + c * step_x
                    y = y0 + r * step_y

                    # --- едем ---
                    DEMO.update(row=r, col=c, phase="moving")
                    try:
                        grbl.move_abs(x, y, feed)
                        grbl.wait_idle(timeout=120)
                    except Exception as e:
                        DEMO["message"] = f"Move error: {e}"

                    # --- стабилизация ---
                    DEMO.update(phase="settling")
                    # settle — ждём указанное время; шаг минимальный 0.05с
                    t_end = time.time() + max(0.05, settle)
                    while DEMO["running"] and time.time() < t_end:
                        time.sleep(0.02)

                    if not DEMO["running"]:
                        return

                    # --- «снимок»: одна короткая вспышка ---
                    DEMO.update(phase="shooting", tick=DEMO["tick"] + 1,
                                shot_at=time.time())
                    # короткая пауза, чтобы UI успел отрисовать вспышку
                    time.sleep(0.12)

    finally:
        DEMO.update(running=False, phase="idle", message="Demo stopped")

# ======================= FastAPI =======================
app = FastAPI(title="PCB Microscope Scanner")
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class MoveIn(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    feed: float = 3000.0


class AdjustIn(BaseModel):
    brightness: Optional[int] = None
    contrast: Optional[int] = None
    saturation: Optional[int] = None


class ScanIn(BaseModel):
    rows: int
    cols: int
    step_x: float
    step_y: float
    feed: float = 3000.0
    snake: bool = True
    settle: float = 0.4


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/captures/{name}")
def get_capture(name: str):
    if STATE["session_dir"] is None:
        raise HTTPException(404, "No session")
    p = STATE["session_dir"] / name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/ports")
def api_ports():
    return [{"device": p.device, "description": p.description}
            for p in list_ports.comports()]


# ---------- GRBL ----------
@app.post("/api/grbl/connect")
def api_grbl_connect(payload: dict = Body(default={})):
    port = payload.get("port") or "AUTO"

    if STATE["grbl"] is not None:
        if port == "AUTO" or port == getattr(STATE["grbl"], "port", None):
            return {"ok": True, "port": STATE["grbl"].port, "note": "already connected"}
        STATE["grbl"].close()
        STATE["grbl"] = None

    if port == "AUTO":
        cands = [p.device for p in list_ports.comports()
                 if any(k in (p.description or "").lower()
                        for k in ("ch340", "cp210", "usb", "serial", "uart"))]
        if not cands:
            raise HTTPException(400, "No serial ports found")
        port = cands[0]

    try:
        ctrl = GRBLController(port, GRBL_BAUD)
        STATE["grbl"] = ctrl
    except Exception as e:
        raise HTTPException(500, f"Open {port} failed: {e}")
    return {"ok": True, "port": port}


@app.post("/api/grbl/disconnect")
def api_grbl_disconnect():
    if STATE["grbl"]:
        STATE["grbl"].close()
        STATE["grbl"] = None
    return {"ok": True}


@app.get("/api/grbl/status")
def api_grbl_status():
    g: Optional[GRBLController] = STATE["grbl"]
    if not g:
        return {"connected": False}
    return {"connected": True, **g.status}


@app.post("/api/grbl/unlock")
def api_grbl_unlock():
    g = STATE["grbl"]
    if not g:
        raise HTTPException(400, "GRBL not connected")
    g.unlock()
    return {"ok": True}


@app.post("/api/grbl/move")
def api_grbl_move(m: MoveIn):
    g = STATE["grbl"]
    if not g:
        raise HTTPException(400, "GRBL not connected")
    try:
        g.move_rel(m.dx, m.dy, m.feed)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True, "status": g.status}


@app.post("/api/grbl/home")
def api_grbl_home():
    g = STATE["grbl"]
    if not g:
        raise HTTPException(400, "GRBL not connected")
    try:
        g.home()
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


@app.post("/api/grbl/goto")
def api_grbl_goto(payload: dict = Body(...)):
    g = STATE["grbl"]
    if not g:
        raise HTTPException(400, "GRBL not connected")
    try:
        g.move_abs(float(payload["x"]), float(payload["y"]),
                   float(payload.get("feed", 3000)))
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


# ---------- Camera / video ----------
@app.post("/api/camera/connect")
def api_cam_connect(payload: dict = Body(default={})):
    idx = int(payload.get("index", CAMERA_INDEX))
    if STATE["camera"]:
        STATE["camera"].close()
        STATE["camera"] = None
    try:
        STATE["camera"] = Camera(idx, CAM_WIDTH, CAM_HEIGHT)
    except Exception as e:
        raise HTTPException(500, f"Camera index {idx} failed: {e}")
    return {"ok": True, "index": idx}


@app.post("/api/camera/disconnect")
def api_cam_disconnect():
    if STATE["camera"]:
        STATE["camera"].close()
        STATE["camera"] = None
    return {"ok": True}


@app.get("/video_feed")
def video_feed():
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            cam: Optional[Camera] = STATE["camera"]
            if cam is None:
                time.sleep(0.2); continue
            frame = cam.get()
            if frame is None:
                time.sleep(0.05); continue
            frame = apply_adjust(frame, STATE["brightness"],
                                 STATE["contrast"], STATE["saturation"])
            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue
            yield boundary + jpg.tobytes() + b"\r\n"
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/adjust")
def api_adjust(a: AdjustIn):
    for k in ("brightness", "contrast", "saturation"):
        v = getattr(a, k)
        if v is not None:
            STATE[k] = int(max(-100, min(100, v)))
    return {k: STATE[k] for k in ("brightness", "contrast", "saturation")}

class NamingIn(BaseModel):
    naming: str  # "coords" | "seq"


@app.get("/api/naming")
def api_naming_get():
    return {"naming": STATE.get("naming", "coords")}


@app.post("/api/naming")
def api_naming_set(n: NamingIn):
    if n.naming not in ("coords", "seq"):
        raise HTTPException(400, "naming must be 'coords' or 'seq'")
    STATE["naming"] = n.naming
    return {"ok": True, "naming": STATE["naming"]}

@app.post("/api/capture")
def api_capture():
    cam: Optional[Camera] = STATE["camera"]
    if not cam:
        raise HTTPException(400, "Camera not connected")
    frame = cam.get()
    if frame is None:
        raise HTTPException(503, "No frame")
    frame = apply_adjust(frame, STATE["brightness"], STATE["contrast"], STATE["saturation"])
    if STATE["session_dir"] is None:
        new_session()
    idx = len(STATE["captures"])
    cols = max(1, STATE["grid_cols"] or 1)
    row, col = idx // cols, idx % cols
    fname = make_filename(row, col, tag=f"manual_{idx:04d}")
    cv2.imwrite(str(STATE["session_dir"] / fname), frame)
    STATE["captures"].append({"row": row, "col": col, "file": fname})
    return {"ok": True, "file": fname, "count": len(STATE["captures"])}


@app.get("/api/captures")
def api_captures():
    return {
        "rows": STATE["grid_rows"],
        "cols": STATE["grid_cols"],
        "captures": STATE["captures"],
        "scan": SCAN,
        "demo": DEMO,
        "naming": STATE.get("naming", "coords"),
    }


@app.post("/api/captures/clear")
def api_captures_clear():
    STATE["captures"] = []
    STATE["session_dir"] = None
    STATE["grid_rows"] = 0
    STATE["grid_cols"] = 0
    return {"ok": True}


# ---------- Scan ----------
@app.post("/api/scan/start")
def api_scan_start(s: ScanIn):
    if SCAN["running"]:
        raise HTTPException(400, "Scan already running")
    if STATE["grbl"] is None:
        raise HTTPException(400, "GRBL not connected")
    if STATE["camera"] is None:
        raise HTTPException(400, "Camera not connected")
    t = threading.Thread(target=run_scan, kwargs=dict(
        rows=s.rows, cols=s.cols, step_x=s.step_x, step_y=s.step_y,
        feed=s.feed, snake=s.snake, settle=s.settle), daemon=True)
    t.start()
    return {"ok": True}


@app.post("/api/scan/stop")
def api_scan_stop():
    SCAN["running"] = False
    return {"ok": True}

@app.post("/api/demo/start")
def api_demo_start(s: ScanIn):
    if DEMO["running"]:
        raise HTTPException(400, "Demo already running")
    if STATE["grbl"] is None:
        raise HTTPException(400, "GRBL not connected")
    if STATE["camera"] is None:
        raise HTTPException(400, "Camera not connected")
    t = threading.Thread(target=run_demo, kwargs=dict(
        rows=s.rows, cols=s.cols, step_x=s.step_x, step_y=s.step_y,
        feed=s.feed, snake=s.snake, settle=s.settle), daemon=True)
    t.start()
    return {"ok": True}


@app.post("/api/demo/stop")
def api_demo_stop():
    DEMO["running"] = False
    return {"ok": True}


@app.get("/api/demo/state")
def api_demo_state():
    return {
        "running": DEMO["running"],
        "row": DEMO["row"],
        "col": DEMO["col"],
        "cycle": DEMO["cycle"],
        "tick": DEMO["tick"],
        "phase": DEMO["phase"],
        "shot_at": DEMO["shot_at"],
        "message": DEMO["message"],
        "rows": STATE["grid_rows"],
        "cols": STATE["grid_cols"],
    }

# ======================= UI =======================
HTML_PAGE = r"""
<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>PCB Microscope Scanner</title>
<style>
*{box-sizing:border-box}
body{margin:0;font:13px/1.4 system-ui,sans-serif;background:#121212;color:#e6e6e6;display:flex;height:100vh;overflow:hidden}
#left{width:340px;padding:10px;overflow-y:auto;background:#1E1E1E;border-right:1px solid #2a2a2e}
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#view{flex:1;display:flex;align-items:center;justify-content:center;background:#000;overflow:hidden;position:relative}
#view img{max-width:100%;max-height:100%;object-fit:contain}
#view .flash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none;transition:opacity .2s ease-out}
#view .flash.on{opacity:.55;transition:opacity .04s ease-in}
#bottom{height:42%;overflow:auto;background:#0b0b0b;border-top:1px solid #2a2a2e;position:relative}
h3{margin:14px 0 6px;font-size:12px;text-transform:uppercase;color:#FFAF26;letter-spacing:.5px}
.row{display:flex;gap:6px;margin-bottom:6px;align-items:center}
.row label{width:60px;color:#9a9a9a}
input,select,button{background:#232327;color:#e6e6e6;border:1px solid #333;padding:5px 7px;border-radius:4px;font:inherit}
input[type=range]{padding:0;accent-color:#FFAF26}
button{cursor:pointer;background:#2a2a30}
button:hover{background:#3a3a44}
button.primary{background:#FFAF26;border-color:#FFAF26;color:#121212;font-weight:600}
button.primary:hover{background:#ffc457}
button.danger{background:#7a2222;border-color:#7a2222}
button.demo{background:#3a2a00;border-color:#FFAF26;color:#FFAF26}
button.demo:hover{background:#4a3600}
.jog-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin:4px 0}
.jog-grid button{padding:8px 4px}
.status{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:#FFAF26;background:#0c0c0e;padding:6px;border-radius:4px;border:1px solid #222}
#grid-inner{display:grid;gap:0;line-height:0}
#grid-inner img{width:100%;display:block}
#grid-inner .cell{aspect-ratio:4/3;background:#111;position:relative;overflow:hidden}
#grid-inner .cell.demo{outline:1px solid #FFAF26}
/* Вспышка-«затвор» внутри ячейки: срабатывает один раз при смене .cell.demo на .flash */
#grid-inner .cell.flash::after{
  content:"";position:absolute;inset:0;background:#fff;opacity:0;
  animation:cellflash .25s ease-out;
}
@keyframes cellflash{0%{opacity:.9}100%{opacity:0}}
#scan-progress{color:#FFAF26;font-size:12px;margin-top:4px}
.small{font-size:11px;color:#666}
#logo{display:flex;align-items:center;gap:8px;margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #2a2a2e}
#logo .logo-img{width:32px;height:32px;flex:0 0 32px;object-fit:contain;display:block}
#logo .name{font-weight:700;color:#FFAF26;letter-spacing:.5px}
#logo .sub{font-size:10px;color:#8a8a92;text-transform:uppercase;letter-spacing:1px}
</style></head><body>

<div id="left">
<div id="logo">
  <img src="/static/logo.svg" alt="PCB Microscope" class="logo-img">
  <div>
    <div class="name">PCB Microscope</div>
    <div class="sub">Scanner</div>
  </div>
</div>
  <h3>GRBL</h3>
  <div class="row">
    <select id="ports" style="flex:1"><option value="AUTO">AUTO</option></select>
    <button onclick="loadPorts()">↻</button>
  </div>
  <div class="row">
    <button class="primary" onclick="grblConnect()" style="flex:1">Connect</button>
    <button onclick="grblDisconnect()">Disconnect</button>
  </div>
  <div class="row">
    <button onclick="grblHome()">$H Home</button>
    <button onclick="grblUnlock()">$X Unlock</button>
  </div>
  <div class="status" id="grbl-status">disconnected</div>

  <h3>Jog</h3>
  <div class="row"><label>Step mm</label>
    <input id="step" type="number" value="1" step="0.1" style="flex:1"></div>
  <div class="row"><label>Feed</label>
    <input id="feed" type="number" value="3000" style="flex:1"></div>
  <div class="jog-grid">
    <button onclick="jog(-1,1)">↖</button><button onclick="jog(0,1)">↑</button><button onclick="jog(1,1)">↗</button>
    <button onclick="jog(-1,0)">←</button><button onclick="jog(0,0)">·</button><button onclick="jog(1,0)">→</button>
    <button onclick="jog(-1,-1)">↙</button><button onclick="jog(0,-1)">↓</button><button onclick="jog(1,-1)">↘</button>
  </div>

  <h3>Camera</h3>
  <div class="row">
    <input id="camidx" type="number" value="0" style="width:60px">
    <button class="primary" onclick="camConnect()" style="flex:1">Connect camera</button>
    <button onclick="camDisconnect()">×</button>
  </div>
  <div class="row"><label>Bright</label><input id="brightness" type="range" min="-100" max="100" value="0" style="flex:1" oninput="setAdjust()"></div>
  <div class="row"><label>Contrast</label><input id="contrast" type="range" min="-100" max="100" value="0" style="flex:1" oninput="setAdjust()"></div>
  <div class="row"><label>Satur</label><input id="saturation" type="range" min="-100" max="100" value="0" style="flex:1" oninput="setAdjust()"></div>
  <div class="row"><button onclick="captureOne()" style="flex:1">📸 Capture</button></div>

  <h3>Auto Scan</h3>
  
  <div class="row"><label>Rows</label><input id="rows" type="number" value="3" style="flex:1"></div>
  <div class="row"><label>Cols</label><input id="cols" type="number" value="5" style="flex:1"></div>
  <div class="row"><label>Step X</label><input id="sx" type="number" value="5" step="0.1" style="flex:1"></div>
  <div class="row"><label>Step Y</label><input id="sy" type="number" value="4" step="0.1" style="flex:1"></div>
  <div class="row"><label>Settle s</label><input id="settle" type="number" value="0.4" step="0.1" style="flex:1"></div>
  <div class="row"><input id="snake" type="checkbox" checked> <label style="width:auto">Snake pattern</label></div>
  <div class="row">
    <button class="primary" onclick="startScan()" style="flex:1">▶ Start</button>
    <button class="danger" onclick="stopScan()">■</button>
  </div>
  <div class="row">
    <button class="demo" onclick="startDemo()" style="flex:1">▶ Demo</button>
    <button class="danger" onclick="stopDemo()">■</button>
  </div>
  <div id="scan-progress"></div>

  <h3>Captures</h3>
  <div class="row">
    <label style="width:auto">Naming</label>
    <select id="naming" onchange="setNaming()" style="flex:1">
      <option value="coords">Координаты (r000_c000)</option>
      <option value="seq">Порядковый номер (0000)</option>
    </select>
  </div>
  <div class="row">
    <button onclick="refreshCaptures()" style="flex:1">Refresh</button>
    <button class="danger" onclick="clearCaptures()">Clear</button>
  </div>
</div>

<div id="main">
  <div id="view"><img id="live" src="/video_feed" alt=""><div class="flash" id="flash"></div></div>
  <div id="bottom"><div id="grid-inner"></div></div>
</div>

<script>
const $ = s => document.querySelector(s);

async function jpost(url, body) {
  const r = await fetch(url, {method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(body||{})});
  if (!r.ok) {
    const txt = await r.text();
    console.error(url, r.status, txt);
    alert(`${url} → ${r.status}\n${txt}`);
    return null;
  }
  return r.json().catch(()=>({}));
}

async function loadPorts() {
  const r = await fetch("/api/ports").then(r=>r.json());
  const sel = $("#ports");
  const cur = sel.value;
  sel.innerHTML = '<option value="AUTO">AUTO</option>' +
    r.map(p=>`<option value="${p.device}">${p.device} — ${p.description||""}</option>`).join("");
  if (cur && [...sel.options].some(o=>o.value===cur)) sel.value = cur;
}

async function grblConnect() {
  const port = $("#ports").value;
  const r = await jpost("/api/grbl/connect", {port});
  if (r) await loadPorts();
}
async function grblDisconnect() { await jpost("/api/grbl/disconnect"); }
async function grblHome() { await jpost("/api/grbl/home"); }
async function grblUnlock() { await jpost("/api/grbl/unlock"); }

async function jog(sx, sy) {
  const step = parseFloat($("#step").value)||1;
  const feed = parseFloat($("#feed").value)||3000;
  await jpost("/api/grbl/move", {dx: sx*step, dy: sy*step, feed});
}

async function camConnect() {
  await jpost("/api/camera/connect", {index: parseInt($("#camidx").value)});
}
async function camDisconnect() { await jpost("/api/camera/disconnect"); }

let adjustTimer = null;
function setAdjust() {
  clearTimeout(adjustTimer);
  adjustTimer = setTimeout(()=> jpost("/api/adjust", {
    brightness: parseInt($("#brightness").value),
    contrast:   parseInt($("#contrast").value),
    saturation: parseInt($("#saturation").value),
  }), 60);
}

async function captureOne() { await jpost("/api/capture"); refreshCaptures(); }

async function startScan() {
  await jpost("/api/scan/start", {
    rows:   parseInt($("#rows").value),
    cols:   parseInt($("#cols").value),
    step_x: parseFloat($("#sx").value),
    step_y: parseFloat($("#sy").value),
    feed:   parseFloat($("#feed").value),
    snake:  $("#snake").checked,
    settle: parseFloat($("#settle").value),
  });
}

async function stopScan() { await jpost("/api/scan/stop"); }

async function clearCaptures() { await jpost("/api/captures/clear"); refreshCaptures(); }

async function loadNaming() {
  try {
    const r = await fetch("/api/naming").then(r=>r.json());
    const sel = $("#naming");
    if (sel && r && r.naming) sel.value = r.naming;
  } catch(e) {}
}

async function setNaming() {
  const naming = $("#naming").value;
  await jpost("/api/naming", { naming });
}

let demoTimer = null;
let lastShotAt = 0;     // последний обработанный shot_at
let lastTick   = 0;     // последний обработанный tick

async function startDemo() {
  await jpost("/api/demo/start", {
    rows:   parseInt($("#rows").value),
    cols:   parseInt($("#cols").value),
    step_x: parseFloat($("#sx").value),
    step_y: parseFloat($("#sy").value),
    feed:   parseFloat($("#feed").value),
    snake:  $("#snake").checked,
    settle: parseFloat($("#settle").value),
  });
  lastShotAt = 0;
  lastTick = 0;
  startDemoPolling();
}

async function stopDemo() {
  await jpost("/api/demo/stop");
  stopDemoPolling();
}

function triggerFlash() {
  const flash = $("#flash");
  flash.classList.remove("on");
  // force reflow, чтобы анимация перезапустилась, даже если шла
  void flash.offsetWidth;
  flash.classList.add("on");
  setTimeout(()=>flash.classList.remove("on"), 140);
}

function startDemoPolling() {
  if (demoTimer) return;
  demoTimer = setInterval(async () => {
    try {
      const d = await fetch("/api/demo/state").then(r=>r.json());
      if (!d.running) {
        stopDemoPolling();
        refreshCaptures();
        return;
      }
      // Реагируем ровно один раз на каждый новый «снимок»
      if (d.tick !== lastTick) {
        lastTick = d.tick;
        lastShotAt = d.shot_at;
        triggerFlash();
      }
    } catch(e){}
  }, 120);
}

function stopDemoPolling() {
  if (demoTimer) { clearInterval(demoTimer); demoTimer = null; }
  const flash = $("#flash");
  flash.classList.remove("on");
}

async function refreshCaptures() {
  const data = await fetch("/api/captures").then(r=>r.json());
  const inner = $("#grid-inner");

  let rows = data.rows, cols = data.cols;
  if (!cols || !rows) {
    const n = data.captures.length;
    cols = Math.ceil(Math.sqrt(n)) || 1;
    rows = Math.ceil(n / cols) || 1;
  }
  inner.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

  const cells = new Array(rows*cols).fill(null);
  for (const c of data.captures) {
    const idx = c.row * cols + c.col;
    if (idx < cells.length) cells[idx] = c;
  }

  const demo = data.demo || {running:false, row:0, col:0, tick:0};
  const demoIdx = demo.running ? (demo.row * cols + demo.col) : -1;

  inner.innerHTML = cells.map((c, i) => {
    if (c) return `<img src="/captures/${c.file}?t=${Date.now()}" alt="">`;
    if (i === demoIdx) {
      const flashCls = (demo.tick && demo.tick === lastTick) ? " flash" : "";
      return `<div class="cell demo${flashCls}"></div>`;
    }
    return `<div class="cell"></div>`;
  }).join("");

  const prog = $("#scan-progress");
  if (data.scan && data.scan.running) {
    prog.textContent = `⏳ ${data.scan.message} (${data.scan.progress}/${data.scan.total})`;
  } else if (demo.running) {
    prog.textContent = `🎬 DEMO  проход ${demo.cycle}  r${demo.row} c${demo.col}  (${demo.phase})`;
  } else if (data.scan && data.scan.message) {
    prog.textContent = data.scan.message;
  } else {
    prog.textContent = "";
  }
}

async function pollStatus() {
  try {
    const s = await fetch("/api/grbl/status").then(r=>r.json());
    if (s.connected) {
      const x = (s.x ?? 0).toFixed(3);
      const y = (s.y ?? 0).toFixed(3);
      $("#grbl-status").textContent = `${s.state}  X:${x}  Y:${y}`;
    } else {
      $("#grbl-status").textContent = "disconnected";
    }
  } catch(e) {}
}

setInterval(pollStatus, 700);
setInterval(refreshCaptures, 1500);
loadPorts();
loadNaming();
refreshCaptures();
</script>
</body></html>
"""
