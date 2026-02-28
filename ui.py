"""
HexaReach — 6-DOF Robotic Arm Control Dashboard
================================================
KPIT Sparkle 2026  |  Team HexaReach

3-column layout:
  LEFT   25%  — Target Input  (sliders, orientation, buttons)
  CENTER 45%  — 3D Viewport   (top) + EE Readback (bottom)
  RIGHT  30%  — Telemetry     (joints, accuracy, solver, geometry, limits)
"""

import time, math
import numpy as np
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QGridLayout,
    QSlider, QSizePolicy, QFrame, QCheckBox, QLineEdit,
    QSplitter,
)
from PyQt6.QtCore import QTimer, Qt, QRectF, QPoint, QSize
from PyQt6.QtGui  import (
    QFont, QImage, QPixmap, QColor, QPainter, QPen, QBrush,
    QLinearGradient,
)

from simulation_threaded import ThreadedRobotSimulation
from gui_control_nn      import NNController
from kinematics          import (HOME_CONFIG, JOINT_LIMITS,
                                 forward_kinematics, log_SO3)


# ─────────────────────────────────────────────────────────────────────────────
#  PALETTE  — HexaReach Dark Aerospace
# ─────────────────────────────────────────────────────────────────────────────
C_BG    = "#0A0E1A"
C_PAN   = "#0D1526"
C_PAN2  = "#111E35"
C_BORD  = "#1A2840"
C_CYAN  = "#00E5FF"
C_AMBER = "#FFB300"
C_NEON  = "#39FF14"
C_WHITE = "#F0F4FF"
C_GREY  = "#8A9BB8"
C_GRID  = "#152030"
C_RED   = "#FF3D55"
C_CYAN_DIM   = "rgba(0,229,255,0.18)"
C_CYAN_MED   = "rgba(0,229,255,0.35)"

FS = 14     # base font size px
FS_MONO = 13

FONT_LABEL = "'Segoe UI Semibold','Arial',sans-serif"
FONT_MONO  = "'JetBrains Mono','Consolas','Courier New',monospace"
FONT_HEAD  = "'Rajdhani','Arial Black','Arial',sans-serif"

DARK_STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {C_BG};
    color: {C_WHITE};
    font-family: {FONT_LABEL};
    font-size: {FS}px;
}}
QSplitter::handle            {{ background: {C_BORD}; }}
QSplitter::handle:horizontal {{ width:  4px; }}
QSplitter::handle:vertical   {{ height: 4px; }}
QGroupBox {{
    background: {C_PAN};
    border: 1px solid {C_BORD};
    border-left: 2px solid {C_CYAN};
    border-radius: 0;
    margin-top: 26px;
    padding: 4px 8px 8px 8px;
    font-family: {FONT_HEAD};
    font-weight: 700;
    font-size: 11px;
    color: {C_CYAN};
    letter-spacing: 2px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 2px 8px;
    left: 6px;
    top: 2px;
    background: {C_PAN};
}}
QSlider::groove:horizontal {{
    height: 3px;
    background: {C_GRID};
    border: none;
}}
QSlider::handle:horizontal {{
    background: {C_CYAN};
    width: 14px; height: 14px;
    margin: -5.5px 0;
    border-radius: 7px;
    border: 2px solid {C_BG};
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 rgba(0,229,255,0.30), stop:1 {C_CYAN});
}}
QLineEdit {{
    background: {C_PAN2};
    color: {C_CYAN};
    border: 1px solid {C_BORD};
    border-bottom: 1px solid rgba(0,229,255,0.55);
    border-radius: 0;
    padding: 2px 4px;
    font-family: {FONT_MONO};
    font-size: {FS_MONO}px;
}}
QLineEdit:focus {{
    border: 1px solid {C_CYAN};
    background: rgba(0,229,255,0.06);
}}
QPushButton {{
    background: transparent;
    color: {C_WHITE};
    border: 1px solid {C_BORD};
    border-radius: 2px;
    padding: 6px 14px;
    font-family: {FONT_LABEL};
    font-weight: 700;
    font-size: {FS}px;
    letter-spacing: 1px;
}}
QPushButton:hover {{
    border-color: {C_CYAN};
    color: {C_CYAN};
    background: rgba(0,229,255,0.07);
}}
QPushButton:pressed {{
    background: rgba(0,229,255,0.18);
}}
QCheckBox {{
    color: {C_GREY};
    spacing: 8px;
    font-size: {FS}px;
}}
QCheckBox::indicator {{
    width:16px; height:16px;
    border: 1px solid {C_BORD};
    background: {C_PAN2};
    border-radius: 1px;
}}
QCheckBox::indicator:checked {{
    background: rgba(0,229,255,0.28);
    border-color: {C_CYAN};
}}
"""

# Slider ranges (restricted to reachable/safe region)
X_MIN, X_MAX   = 100, 600
Y_MIN, Y_MAX   = -400, 400
Z_MIN, Z_MAX   = 150, 700
RL_MIN, RL_MAX = -180, 180
PT_MIN, PT_MAX =  -60,  60
YW_MIN, YW_MAX = -180, 180


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _sep(v=False):
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine if v else QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{C_BORD};")
    return f


def _lbl(text, color=C_WHITE, bold=False, size=FS, mono=False,
         align=Qt.AlignmentFlag.AlignLeft):
    w = QLabel(text)
    ff = FONT_MONO if mono else FONT_LABEL
    st = f"color:{color}; font-size:{size}px; font-family:{ff};"
    if bold:
        st += " font-weight:700;"
    w.setStyleSheet(st)
    w.setAlignment(align)
    return w


def _cyan_title(text):
    """Section sub-title."""
    w = QLabel(text.upper())
    w.setStyleSheet(
        f"color:{C_CYAN}; font-size:10px; font-family:{FONT_HEAD};"
        f"font-weight:700; letter-spacing:2px; padding:2px 0;")
    return w


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
class PulseDot(QLabel):
    """Animated ● indicator that slowly pulses."""
    def __init__(self, col_on=C_NEON, col_off="#143010", size=13, parent=None):
        super().__init__("●", parent)
        self._on  = col_on
        self._off = col_off
        self._sz  = size
        self._bright = True
        self.setStyleSheet(f"color:{col_on}; font-size:{size}px;")
        t = QTimer(self);  t.timeout.connect(self._blink);  t.start(900)

    def _blink(self):
        self._bright = not self._bright
        c = self._on if self._bright else self._off
        self.setStyleSheet(f"color:{c}; font-size:{self._sz}px;")

    def set_ok(self, ok: bool):
        self._on  = C_NEON if ok else C_RED
        self._off = "#143010" if ok else "#3A0E0E"


class AccuracyGauge(QWidget):
    """Circular donut gauge drawn with QPainter."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._val = 0.0
        self.setFixedSize(88, 88)

    def set_value(self, v: float):
        self._val = max(0.0, min(100.0, float(v)))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        m   = 9
        rect = QRectF(m, m, self.width()-2*m, self.height()-2*m)

        pen = QPen(QColor(C_GRID), 8)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        p.setPen(pen); p.drawArc(rect, 0, 360*16)

        c = QColor(C_NEON) if self._val >= 99 else (
            QColor(C_AMBER) if self._val >= 90 else QColor(C_RED))
        pen = QPen(c, 8); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawArc(rect, 90*16, int(-self._val / 100.0 * 360 * 16))

        p.setPen(QPen(QColor(C_WHITE)))
        font = QFont("Consolas", 10, QFont.Weight.Bold)
        p.setFont(font)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                   f"{self._val:.1f}%")


class JointBar(QWidget):
    """Thin bar: position of current angle within joint limit range."""
    def __init__(self, lo_deg: float, hi_deg: float, parent=None):
        super().__init__(parent)
        self._lo  = lo_deg
        self._hi  = hi_deg
        self._val = 0.0
        self.setFixedHeight(4)
        self.setMinimumWidth(40)

    def set_value(self, deg: float):
        self._val = float(deg)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(C_GRID))
        r = (self._val - self._lo) / max(self._hi - self._lo, 1.0)
        r = max(0.0, min(1.0, r))
        col = QColor(C_CYAN) if self._val >= 0 else QColor(C_AMBER)
        fw = max(2, int(r * w))
        p.fillRect(0, 0, fw, h, col)
        tx = max(0, min(w - 3, int(r * w) - 1))
        p.fillRect(tx, 0, 3, h, QColor(C_WHITE))


class ViewportLabel(QLabel):
    """Interactive 3D render target — drag orbit, scroll zoom."""
    def __init__(self, sim, parent=None):
        super().__init__(parent)
        self._sim = sim
        self._drag = None
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background:{C_GRID}; border:1px solid {C_BORD};")
        self.setText("▶  Initialising 3D render…")
        # DO NOT use setScaledContents(True) — causes Qt bilinear upscaling blur.
        # We render at native physical-pixel resolution so no scaling is needed.
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setMinimumHeight(240)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,
                           QSizePolicy.Policy.Ignored)

    def sizeHint(self):
        return QSize(0, 0)

    # -- tell the background render thread the exact physical pixel size ----------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        dpr = self.devicePixelRatio()
        pw  = max(64,  int(self.width()  * dpr))
        ph  = max(36,  int(self.height() * dpr))
        self._sim.set_render_size(pw, ph)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag:
            d = e.position().toPoint() - self._drag
            self._sim.orbit_camera(d.x() * 0.4, d.y() * 0.4)
            self._drag = e.position().toPoint()

    def mouseReleaseEvent(self, _):
        self._drag = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def wheelEvent(self, e):
        self._sim.zoom_camera(-e.angleDelta().y() * 0.001)


# ─────────────────────────────────────────────────────────────────────────────
#  SLIDER ROW
# ─────────────────────────────────────────────────────────────────────────────
class SliderRow:
    def __init__(self, name, lo, hi, init, unit="mm", color=C_CYAN):
        self.lo = lo; self.hi = hi; self._busy = False

        self.name_lbl = QLabel(f"<b>{name}</b>")
        self.name_lbl.setStyleSheet(
            f"color:{color}; font-size:{FS}px; min-width:46px;"
            f"font-family:{FONT_MONO}; font-weight:700;")

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(lo); self.slider.setMaximum(hi)
        self.slider.setValue(int(init))
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Fixed)
        self.slider.setFixedHeight(28)

        self.edit = QLineEdit(str(init))
        self.edit.setFixedWidth(80)
        self.edit.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.unit_lbl = QLabel(unit)
        self.unit_lbl.setStyleSheet(
            f"color:{C_GREY}; font-size:12px; min-width:26px;")

        self.slider.valueChanged.connect(self._sv)
        self.edit.editingFinished.connect(self._ed)
        self.edit.returnPressed.connect(self._ed)

    def _sv(self, v):
        if self._busy: return
        self._busy = True; self.edit.setText(str(v)); self._busy = False

    def _ed(self):
        if self._busy: return
        try:    v = int(round(float(self.edit.text())))
        except: v = self.slider.value()
        v = int(max(self.lo, min(self.hi, v)))
        self._busy = True
        self.slider.setValue(v); self.edit.setText(str(v))
        self._busy = False

    def value(self): return self.slider.value()

    def set_value(self, v):
        v = int(max(self.lo, min(self.hi, round(float(v)))))
        self._busy = True
        self.slider.setValue(v); self.edit.setText(str(v))
        self._busy = False

    def add_to_grid(self, grid, row):
        grid.addWidget(self.name_lbl, row, 0)
        grid.addWidget(self.slider,   row, 1)
        grid.addWidget(self.edit,     row, 2)
        grid.addWidget(self.unit_lbl, row, 3)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class RobotControlGUI(QMainWindow):

    def __init__(self, sim=None, controller=None):
        super().__init__()
        self.setWindowTitle("HexaReach  |  6-DOF Robotic Arm  |  KPIT Sparkle 2026")
        self.setStyleSheet(DARK_STYLESHEET)

        self.sim = sim or ThreadedRobotSimulation()
        if sim is None: self.sim.start()
        self.controller = controller or NNController(self.sim)

        self._demo_mode = False; self._demo_t = 0.0
        self._frame_times = []; self._tick_num = 0
        self._last_ts = datetime.now().strftime("%H:%M:%S")
        self._prev_fd = [0.0] * 6   # for flash detection

        self._build_ui()

        self._ctrl_tmr = QTimer(); self._ctrl_tmr.timeout.connect(self._tick)
        self._ctrl_tmr.start(33)   # ~30 Hz – lighter on UI thread

        self._rnd_tmr = QTimer(); self._rnd_tmr.timeout.connect(self._render_vp)
        self._rnd_tmr.start(33)    # ~30 Hz – reads cached frame

        self.showMaximized()

    # ─────────────────────────────────────────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root  = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_header())   # 60px

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setChildrenCollapsible(False)
        body.addWidget(self._build_left())
        body.addWidget(self._build_center())
        body.addWidget(self._build_right())
        body.setStretchFactor(0, 25)
        body.setStretchFactor(1, 45)
        body.setStretchFactor(2, 30)
        outer.addWidget(body, stretch=1)

        outer.addWidget(self._build_footer())   # 28px

    # ─── Header ──────────────────────────────────────────────────────────────
    def _build_header(self):
        w = QWidget()
        w.setFixedHeight(58)
        w.setStyleSheet(
            f"background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {C_PAN2}, stop:0.5 #0D1A30, stop:1 {C_PAN2});"
            f"border-bottom: 1px solid rgba(0,229,255,0.20);")
        row = QHBoxLayout(w)
        row.setContentsMargins(14, 4, 14, 4)
        row.setSpacing(0)

        # Logo
        logo = QLabel()
        logo.setText(
            "<span style='color:#00E5FF;font-family:Rajdhani,Arial Black;"
            "font-size:28px;font-weight:900;letter-spacing:-1px;'>HEXA</span>"
            "<span style='color:#F0F4FF;font-family:Rajdhani,Arial Black;"
            "font-size:28px;font-weight:900;'>REACH</span>")
        row.addWidget(logo)

        div = QLabel("  │  ")
        div.setStyleSheet(f"color:{C_BORD}; font-size:22px;")
        row.addWidget(div)

        subtitle = QLabel("6-DOF ROBOTIC ARM  —  INDUSTRY CONTROL SYSTEM")
        subtitle.setStyleSheet(
            f"color:{C_GREY}; font-size:13px; font-family:{FONT_HEAD};"
            f"font-weight:700; letter-spacing:2px;")
        row.addWidget(subtitle)
        row.addStretch()

        # Center status pills
        pill_w = QWidget()
        pill_row = QHBoxLayout(pill_w)
        pill_row.setContentsMargins(0, 0, 0, 0)
        pill_row.setSpacing(6)

        self._pill_online = self._make_pill("● ONLINE", C_NEON)
        self._pill_conv   = self._make_pill("IK CONVERGED", C_NEON)
        self._pill_acc    = self._make_pill("100.00%  ACC", C_NEON)
        for p in (self._pill_online, self._pill_conv, self._pill_acc):
            pill_row.addWidget(p)
        pill_w.setStyleSheet("background:transparent;")
        row.addWidget(pill_w)
        row.addStretch()

        # Right side
        self._fps_lbl   = QLabel("FPS: --")
        self._clock_lbl = QLabel()
        for lb in (self._fps_lbl, self._clock_lbl):
            lb.setStyleSheet(f"color:{C_GREY}; font-size:13px;")
            row.addWidget(lb)

        badge = QLabel("  ✦ KPIT SPARKLE 2026")
        badge.setStyleSheet(
            f"color:{C_AMBER}; font-size:13px; font-family:{FONT_HEAD};"
            f"font-weight:700; letter-spacing:1px;"
            f"border:1px solid rgba(255,179,0,0.4); border-radius:2px;"
            f"padding:2px 8px;")
        row.addWidget(badge)
        return w

    @staticmethod
    def _make_pill(text, color):
        w = QLabel(text)
        w.setStyleSheet(
            f"color:{color}; font-size:12px; font-family:Rajdhani,'Arial',sans-serif;"
            f"font-weight:700; letter-spacing:1px;"
            f"border:1px solid rgba(57,255,20,0.30); border-radius:10px;"
            f"padding:1px 10px; background:rgba(57,255,20,0.06);")
        return w

    # ─── Footer ──────────────────────────────────────────────────────────────
    def _build_footer(self):
        w = QWidget()
        w.setFixedHeight(28)
        w.setStyleSheet(
            f"background:{C_PAN};"
            f"border-top:1px solid rgba(0,229,255,0.12);")
        row = QHBoxLayout(w)
        row.setContentsMargins(14, 2, 14, 2)
        row.setSpacing(18)

        self._badge_reach = QLabel("◉  INITIALISING")
        self._badge_reach.setStyleSheet(
            f"color:{C_GREY}; font-size:13px; font-weight:700;")
        row.addWidget(self._badge_reach)

        self._badge_clamp = QLabel("")
        self._badge_clamp.setStyleSheet(
            f"color:{C_AMBER}; font-size:13px; font-weight:700;")
        row.addWidget(self._badge_clamp)

        row.addStretch()

        mid_lbl = QLabel("HEXAREACH v1.0  ·  KPIT SPARKLE 2026  ·  6-DOF SPHERICAL WRIST")
        mid_lbl.setStyleSheet(
            f"color:rgba(138,155,184,0.55); font-size:11px; letter-spacing:1px;")
        mid_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(mid_lbl)
        row.addStretch()

        self._badge_ts = QLabel("Last valid: --")
        self._badge_ts.setStyleSheet(f"color:{C_GREY}; font-size:12px;")
        row.addWidget(self._badge_ts)
        return w

    # ─── LEFT panel (25%) ────────────────────────────────────────────────────
    def _build_left(self):
        panel = QWidget()
        panel.setStyleSheet(f"background:{C_BG};")
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 4, 8)
        lay.setSpacing(6)

        # TARGET POSITION
        pos_box  = QGroupBox("TARGET POSITION")
        pos_grid = QGridLayout(pos_box)
        pos_grid.setContentsMargins(6, 22, 6, 6)
        pos_grid.setVerticalSpacing(6)
        pos_grid.setColumnStretch(1, 1)
        pos_grid.addWidget(
            _lbl(f"X [{X_MIN}…{X_MAX}]  Y [{Y_MIN}…{Y_MAX}]  Z [{Z_MIN}…{Z_MAX}]",
                 C_GREY, size=11), 0, 0, 1, 4)
        self.sl_x = SliderRow("X", X_MIN, X_MAX, 300, "mm", C_CYAN)
        self.sl_y = SliderRow("Y", Y_MIN, Y_MAX,   0, "mm", C_CYAN)
        self.sl_z = SliderRow("Z", Z_MIN, Z_MAX, 350, "mm", C_CYAN)
        self.sl_x.add_to_grid(pos_grid, 1)
        self.sl_y.add_to_grid(pos_grid, 2)
        self.sl_z.add_to_grid(pos_grid, 3)
        lay.addWidget(pos_box)

        # END-EFFECTOR ORIENTATION
        ori_box  = QGroupBox("END-EFFECTOR ORIENTATION")
        ori_grid = QGridLayout(ori_box)
        ori_grid.setContentsMargins(6, 22, 6, 6)
        ori_grid.setVerticalSpacing(6)
        ori_grid.setColumnStretch(1, 1)
        ori_grid.addWidget(
            _lbl("Roll [±180]   Pitch [-90…+63]   Yaw [±180]",
                 C_GREY, size=11), 0, 0, 1, 4)
        self.sl_roll  = SliderRow("Roll",  RL_MIN, RL_MAX,  0, "°", C_AMBER)
        self.sl_pitch = SliderRow("Pitch", PT_MIN, PT_MAX,  0, "°", C_AMBER)
        self.sl_yaw   = SliderRow("Yaw",   YW_MIN, YW_MAX,  0, "°", C_AMBER)
        self.sl_roll.add_to_grid(ori_grid,  1)
        self.sl_pitch.add_to_grid(ori_grid, 2)
        self.sl_yaw.add_to_grid(ori_grid,   3)
        lay.addWidget(ori_box)

        # CONTROL ACTIONS
        ctrl_box = QGroupBox("CONTROL ACTIONS")
        ctrl_lay = QVBoxLayout(ctrl_box)
        ctrl_lay.setContentsMargins(6, 22, 6, 6)
        ctrl_lay.setSpacing(6)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._btn_home = QPushButton("⟳  RESET HOME")
        self._btn_home.setStyleSheet(
            f"color:{C_CYAN}; border:1px solid rgba(0,229,255,0.5);"
            f"padding:7px 10px; font-size:{FS}px; font-weight:700;")
        self._btn_home.clicked.connect(self._on_home)
        self._btn_rand = QPushButton("⚄  RANDOM POSE")
        self._btn_rand.setStyleSheet(
            f"color:{C_AMBER}; border:1px solid rgba(255,179,0,0.5);"
            f"padding:7px 10px; font-size:{FS}px; font-weight:700;")
        self._btn_rand.clicked.connect(self._on_random)
        btn_row.addWidget(self._btn_home)
        btn_row.addWidget(self._btn_rand)
        ctrl_lay.addLayout(btn_row)

        demo_row = QHBoxLayout()
        demo_row.setSpacing(8)
        self._demo_chk = QCheckBox("▶  Demo Mode  (auto-sweep trajectory)")
        self._demo_chk.toggled.connect(self._on_demo_toggle)
        demo_row.addWidget(self._demo_chk)
        demo_row.addStretch()
        ctrl_lay.addLayout(demo_row)
        lay.addWidget(ctrl_box)

        lay.addStretch(1)
        return panel

    # ─── CENTER panel (45%) ───────────────────────────────────────────────────
    def _build_center(self):
        panel = QWidget()
        panel.setStyleSheet(f"background:{C_BG};")
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # 3D viewport (splitter so user can resize)
        csplit = QSplitter(Qt.Orientation.Vertical)
        csplit.setChildrenCollapsible(False)

        # Viewport section
        vp_box = QGroupBox(
            "PyBullet 3D VIEWPORT   ·   6-DOF | SPHERICAL WRIST | MAX REACH 750mm")
        vp_lay = QVBoxLayout(vp_box)
        vp_lay.setContentsMargins(4, 24, 4, 4)
        vp_lay.setSpacing(0)

        self._viewport = ViewportLabel(self.sim)
        # Corner hint overlay
        hint = QLabel("DRAG TO ORBIT  ·  SCROLL TO ZOOM  ·  PyBullet ENGINE")
        hint.setStyleSheet(
            f"color:rgba(0,229,255,0.35); font-size:10px; letter-spacing:1px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignRight)

        vp_lay.addWidget(self._viewport, stretch=1)
        vp_lay.addWidget(hint)
        csplit.addWidget(vp_box)

        # EE Readback section
        ee_box  = QGroupBox("END-EFFECTOR  FK READBACK")
        ee_lay  = QGridLayout(ee_box)
        ee_lay.setContentsMargins(8, 24, 8, 8)
        ee_lay.setVerticalSpacing(6)
        ee_lay.setColumnStretch(1, 1)

        # Pulse dot + FK match indicator
        dot_row = QHBoxLayout()
        self._ee_dot = PulseDot(C_NEON, "#143010", 14)
        dot_row.addWidget(self._ee_dot)
        dot_row.addWidget(_lbl("Live position data", C_GREY, size=11))
        dot_row.addStretch()
        self._ee_match = _lbl("", C_NEON, bold=True, size=FS)
        dot_row.addWidget(self._ee_match)
        ee_lay.addLayout(dot_row, 0, 0, 1, 2)

        ee_lay.addWidget(_lbl("FK Pos ", C_GREY, size=FS), 1, 0)
        self._lbl_ee_pos = _lbl("--", C_WHITE, bold=True, mono=True, size=FS)
        ee_lay.addWidget(self._lbl_ee_pos, 1, 1)

        ee_lay.addWidget(_lbl("Target  ", C_GREY, size=FS), 2, 0)
        self._lbl_ee_tgt = _lbl("--", C_GREY, mono=True, size=FS)
        ee_lay.addWidget(self._lbl_ee_tgt, 2, 1)
        csplit.addWidget(ee_box)

        csplit.setStretchFactor(0, 4)
        csplit.setStretchFactor(1, 1)
        csplit.setSizes([600, 150])
        lay.addWidget(csplit, stretch=1)
        return panel

    # ─── RIGHT panel (30%) ────────────────────────────────────────────────────
    def _build_right(self):
        panel = QWidget()
        panel.setStyleSheet(f"background:{C_BG};")
        lay   = QVBoxLayout(panel)
        lay.setContentsMargins(4, 8, 8, 8)
        lay.setSpacing(6)

        lay.addWidget(self._build_joint_angles())
        lay.addWidget(self._build_accuracy_solver())
        lay.addWidget(self._build_geo_limits())
        lay.addStretch(1)
        return panel

    def _build_joint_angles(self):
        box  = QGroupBox("JOINT ANGLES")
        grid = QGridLayout(box)
        grid.setContentsMargins(6, 24, 6, 6)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 2); grid.setColumnStretch(2, 2)
        grid.setColumnStretch(3, 1)

        hdrs = [("JOINT", C_GREY), ("REFINED (°)", C_CYAN),
                ("NN SEED (°)", C_GREY), ("%RNG", C_GREY)]
        for c, (t, col) in enumerate(hdrs):
            grid.addWidget(_lbl(t, col, bold=True, size=11), 0, c)

        JLIMS = [(-180,180), (-90,90), (-90,63),
                 (-180,180), (-90,63), (-360,360)]
        self._ja_final = [];  self._ja_seed = [];  self._jbars = []
        for i in range(6):
            lo, hi = JLIMS[i]
            n  = _lbl(f"J{i+1}", C_WHITE, bold=True, size=FS)
            f  = _lbl("--", C_CYAN, bold=True, mono=True, size=FS,
                      align=Qt.AlignmentFlag.AlignRight)
            s  = _lbl("--", C_GREY, mono=True, size=FS,
                      align=Qt.AlignmentFlag.AlignRight)
            b  = JointBar(lo, hi)
            row = i + 1
            grid.addWidget(n, row, 0)
            grid.addWidget(f, row, 1)
            grid.addWidget(s, row, 2)
            grid.addWidget(b, row, 3)
            self._ja_final.append(f)
            self._ja_seed.append(s)
            self._jbars.append(b)
        return box

    def _build_accuracy_solver(self):
        box = QWidget()
        box.setStyleSheet(f"background:transparent;")
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        # ── Accuracy metrics ──────────────────────────────────────────
        acc_box  = QGroupBox("ACCURACY METRICS")
        acc_lay  = QVBoxLayout(acc_box)
        acc_lay.setContentsMargins(6, 24, 6, 6)
        acc_lay.setSpacing(4)

        gauge_row = QHBoxLayout()
        self._gauge = AccuracyGauge()
        gauge_row.addWidget(self._gauge)

        vals = QGridLayout()
        vals.setVerticalSpacing(4)

        def mrow(grid, r, name, init, col=C_WHITE):
            n = _lbl(name, C_GREY, size=FS-1)
            v = _lbl(init, col, bold=True, mono=True, size=FS,
                     align=Qt.AlignmentFlag.AlignRight)
            grid.addWidget(n, r, 0); grid.addWidget(v, r, 1)
            return v

        self._lbl_pos = mrow(vals, 0, "Pos Err",  "--  mm", C_WHITE)
        self._lbl_ori = mrow(vals, 1, "Ori Err",  "--  °",  C_WHITE)
        self._lbl_acc = mrow(vals, 2, "Accuracy", "--  %",  C_NEON)
        gauge_row.addLayout(vals, stretch=1)
        acc_lay.addLayout(gauge_row)
        row.addWidget(acc_box, 1)

        # ── Solver diagnostics ────────────────────────────────────────
        sol_box  = QGroupBox("SOLVER DIAGNOSTICS")
        sol_grid = QGridLayout(sol_box)
        sol_grid.setContentsMargins(6, 24, 6, 6)
        sol_grid.setVerticalSpacing(4)
        sol_grid.setColumnStretch(1, 1)

        def srow(grid, r, name):
            n = _lbl(name, C_GREY, size=FS-1)
            v = _lbl("--", C_WHITE, mono=True, size=FS,
                     align=Qt.AlignmentFlag.AlignRight)
            grid.addWidget(n, r, 0); grid.addWidget(v, r, 1)
            return v

        self._lbl_mode    = srow(sol_grid, 0, "Solver")
        self._lbl_iters   = srow(sol_grid, 1, "DLS Iters")
        self._lbl_ms      = srow(sol_grid, 2, "Solve Time")

        # Converged row with pulse dot
        conv_lbl = _lbl("Converged", C_GREY, size=FS-1)
        conv_row = QHBoxLayout()
        self._conv_dot  = PulseDot(C_NEON, "#143010", 12)
        self._lbl_conv  = _lbl("--", C_NEON, bold=True, size=FS,
                               align=Qt.AlignmentFlag.AlignRight)
        conv_row.addWidget(self._conv_dot)
        conv_row.addStretch()
        conv_row.addWidget(self._lbl_conv)
        sol_grid.addWidget(conv_lbl,          3, 0)
        sol_grid.addLayout(conv_row,          3, 1)

        clamp_lbl = _lbl("Clamped", C_GREY, size=FS-1)
        self._lbl_clamp = _lbl("--", C_GREY, bold=True, size=FS,
                               align=Qt.AlignmentFlag.AlignRight)
        sol_grid.addWidget(clamp_lbl,         4, 0)
        sol_grid.addWidget(self._lbl_clamp,   4, 1)
        row.addWidget(sol_box, 1)
        return box

    def _build_geo_limits(self):
        box = QWidget()
        box.setStyleSheet("background:transparent;")
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        # ── Robot Geometry ────────────────────────────────────────────
        geo_box  = QGroupBox("ROBOT GEOMETRY")
        geo_grid = QGridLayout(geo_box)
        geo_grid.setContentsMargins(6, 24, 6, 6)
        geo_grid.setVerticalSpacing(3)
        geo_data = [
            ("Base Height",       "100 mm"),
            ("Shoulder → Elbow",  "400 mm"),
            ("Elbow → Wrist",     "300 mm [sph.wrist]"),
            ("Wrist → EE (tool)", " 50 mm"),
            ("Max Reach",         "750 mm"),
        ]
        for r, (k, v) in enumerate(geo_data):
            geo_grid.addWidget(_lbl(k, C_GREY,  size=FS-1), r, 0)
            geo_grid.addWidget(_lbl(v, C_WHITE, bold=True, mono=True, size=FS-1,
                align=Qt.AlignmentFlag.AlignRight), r, 1)
        row.addWidget(geo_box, 1)

        # ── Joint Limits ──────────────────────────────────────────────
        lim_box  = QGroupBox("JOINT LIMITS")
        lim_grid = QGridLayout(lim_box)
        lim_grid.setContentsMargins(6, 24, 6, 6)
        lim_grid.setVerticalSpacing(2)
        lim_grid.setColumnStretch(1, 1); lim_grid.setColumnStretch(2, 1)

        lim_data = [
            ("J1 Base Yaw",    "-180°", "+180°"),
            ("J2 Shoulder",    " -90°", " +90°"),
            ("J3 Elbow",       " -90°", " +63°"),
            ("J4 Forearm Roll","-180°", "+180°"),
            ("J5 Wrist Pitch", " -90°", " +63°"),
            ("J6 Wrist Roll",  "-360°", "+360°"),
        ]
        lim_grid.addWidget(_lbl("JOINT",  C_GREY, bold=True, size=11), 0, 0)
        lim_grid.addWidget(_lbl("MIN",    C_AMBER, bold=True, size=11,
            align=Qt.AlignmentFlag.AlignRight), 0, 1)
        lim_grid.addWidget(_lbl("MAX",    C_CYAN, bold=True, size=11,
            align=Qt.AlignmentFlag.AlignRight), 0, 2)
        for r, (j, lo, hi) in enumerate(lim_data, 1):
            lim_grid.addWidget(_lbl(j,  C_GREY,  size=FS-1), r, 0)
            lim_grid.addWidget(_lbl(lo, C_AMBER, mono=True, size=FS-1,
                align=Qt.AlignmentFlag.AlignRight), r, 1)
            lim_grid.addWidget(_lbl(hi, C_CYAN,  mono=True, size=FS-1,
                align=Qt.AlignmentFlag.AlignRight), r, 2)
        row.addWidget(lim_box, 1)
        return box

    # ─────────────────────────────────────────────────────────────────────────
    #  30 Hz CONTROL TICK
    # ─────────────────────────────────────────────────────────────────────────
    def _tick(self):
        t0 = time.perf_counter()
        self._tick_num += 1

        if self._demo_mode:
            self._demo_t += 0.033
            self.sl_x.set_value(int(350 + 250 * np.sin(self._demo_t * 0.35)))
            self.sl_y.set_value(int(0   + 400 * np.sin(self._demo_t * 0.23)))
            self.sl_z.set_value(int(425 + 250 * np.sin(self._demo_t * 0.29)))

        xm = self.sl_x.value(); ym = self.sl_y.value(); zm = self.sl_z.value()
        rl = self.sl_roll.value(); pt = self.sl_pitch.value(); yw = self.sl_yaw.value()
        pos_m = np.array([xm, ym, zm], dtype=float) / 1000.0

        cr,sr = np.cos(np.radians(rl)),  np.sin(np.radians(rl))
        cp,sp = np.cos(np.radians(pt)),  np.sin(np.radians(pt))
        cy,sy = np.cos(np.radians(yw)),  np.sin(np.radians(yw))
        R_tgt = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,             cp*cr           ]
        ])

        r = self.controller.solve_and_apply(pos_m, R_tgt)
        self._upd_joints(r)
        self._upd_errors(r)
        self._upd_solver(r)
        self._upd_status(r)
        self._upd_ee(r["q_final"], r.get("tgt_pos_m", pos_m))

        self._frame_times.append(time.perf_counter() - t0)
        if len(self._frame_times) > 30: self._frame_times.pop(0)
        if self._tick_num % 30 == 0:
            fps = min(999, 1.0 / max(np.mean(self._frame_times), 1e-6))
            self._fps_lbl.setText(f"FPS: {fps:.0f}")
            self._clock_lbl.setText(f"  {datetime.now().strftime('%H:%M:%S')}")

    # ─────────────────────────────────────────────────────────────────────────
    #  25 Hz VIEWPORT RENDER  — reads pre-rendered frame, never blocks
    # ─────────────────────────────────────────────────────────────────────────
    def _render_vp(self):
        rgb = self.sim.get_latest_frame()   # instant – background thread pre-renders
        if rgb is None:
            return
        h_img, w_img, _ = rgb.shape
        # Build QImage from the native-resolution buffer.
        # Dynamically set devicePixelRatio so image fills widget cleanly even if
        # render size is capped for performance on high-DPI displays.
        img = QImage(rgb.tobytes(), w_img, h_img,
                     w_img * 3, QImage.Format.Format_RGB888)
        vw = max(1, self._viewport.width())
        vh = max(1, self._viewport.height())
        sx = w_img / float(vw)
        sy = h_img / float(vh)
        dpr = min(sx, sy) if (sx > 0 and sy > 0) else 1.0
        img.setDevicePixelRatio(max(0.1, dpr))
        # setPixmap without setScaledContents — image already matches physical pixels.
        self._viewport.setPixmap(QPixmap.fromImage(img))
        self._viewport.setText("")   # clear "Initialising" text once first frame arrives

    # ─────────────────────────────────────────────────────────────────────────
    #  DISPLAY UPDATERS
    # ─────────────────────────────────────────────────────────────────────────
    def _upd_joints(self, r):
        JLIMS = [(-180,180), (-90,90), (-90,63),
                 (-180,180), (-90,63), (-360,360)]
        fd = r["q_final_deg"]; sd = r["q_seed_deg"]
        for i in range(6):
            val = fd[i]
            changed = abs(val - self._prev_fd[i]) > 0.05
            # color coding: cyan positive, amber negative, grey zero
            col = (C_CYAN if val > 0.05
                   else C_AMBER if val < -0.05
                   else C_GREY)
            self._ja_final[i].setText(f"{val:+7.2f}°")
            if changed:
                self._ja_final[i].setStyleSheet(
                    f"color:{C_WHITE}; font-size:{FS}px; font-weight:700;"
                    f"font-family:{FONT_MONO};")
                QTimer.singleShot(280, lambda lbl=self._ja_final[i], c=col:
                    lbl.setStyleSheet(
                        f"color:{c}; font-size:{FS}px; font-weight:700;"
                        f"font-family:{FONT_MONO};"))
            else:
                self._ja_final[i].setStyleSheet(
                    f"color:{col}; font-size:{FS}px; font-weight:700;"
                    f"font-family:{FONT_MONO};")
            self._prev_fd[i] = val
            self._jbars[i].set_value(val)
            self._ja_seed[i].setText(f"{sd[i]:+7.2f}°")

    def _upd_errors(self, r):
        if not r.get("reachable", True):
            self._lbl_pos.setText("--")
            self._lbl_pos.setStyleSheet(
                f"color:{C_RED}; font-weight:700; font-size:{FS}px; font-family:{FONT_MONO};")
            self._lbl_ori.setText("--")
            self._lbl_ori.setStyleSheet(
                f"color:{C_RED}; font-weight:700; font-size:{FS}px; font-family:{FONT_MONO};")
            self._lbl_acc.setText("--")
            self._lbl_acc.setStyleSheet(
                f"color:{C_RED}; font-weight:900; font-size:{FS+4}px;"
                f"font-family:{FONT_MONO};")
            self._gauge.set_value(0.0)
            self._pill_acc.setText("UNREACHABLE")
            self._pill_acc.setStyleSheet(
                f"color:{C_RED}; font-size:12px; font-family:Rajdhani,'Arial',sans-serif;"
                f"font-weight:700; letter-spacing:1px;"
                f"border:1px solid rgba(255,61,85,0.35); border-radius:10px;"
                f"padding:1px 10px; background:rgba(255,61,85,0.06);")
            return

        em = r["pos_error_mm"]; ed = r["ori_error_deg"]; ac = r["accuracy"]
        pc  = C_NEON  if em <= 1.0 else (C_AMBER if em <= 5.0 else C_RED)
        oc  = C_NEON  if ed <= 1.0 else (C_AMBER if ed <= 5.0 else C_RED)
        ac_c = C_NEON if ac >= 99.0 else (C_AMBER if ac >= 90.0 else C_RED)

        self._lbl_pos.setText(f"{em:8.4f} mm")
        self._lbl_pos.setStyleSheet(
            f"color:{pc}; font-weight:700; font-size:{FS}px; font-family:{FONT_MONO};")
        self._lbl_ori.setText(f"{ed:8.4f} °")
        self._lbl_ori.setStyleSheet(
            f"color:{oc}; font-weight:700; font-size:{FS}px; font-family:{FONT_MONO};")
        self._lbl_acc.setText(f"{ac:6.2f} %")
        self._lbl_acc.setStyleSheet(
            f"color:{ac_c}; font-weight:900; font-size:{FS+4}px;"
            f"font-family:{FONT_MONO};")
        self._gauge.set_value(ac)

        # Update header accuracy pill
        self._pill_acc.setText(f"{ac:.2f}%  ACC")
        acc_col = "#39FF14" if ac >= 99 else "#FFB300"
        self._pill_acc.setStyleSheet(
            f"color:{acc_col}; font-size:12px; font-family:Rajdhani,'Arial',sans-serif;"
            f"font-weight:700; letter-spacing:1px;"
            f"border:1px solid rgba(57,255,20,0.30); border-radius:10px;"
            f"padding:1px 10px; background:rgba(57,255,20,0.06);")

    def _upd_solver(self, r):
        if not r.get("reachable", True):
            self._lbl_mode.setText("—")
            self._lbl_mode.setStyleSheet(f"color:{C_RED}; font-size:{FS}px;")
            self._lbl_iters.setText("--")
            self._lbl_ms.setText("--")
            self._lbl_conv.setText("OUT OF RANGE")
            self._lbl_conv.setStyleSheet(
                f"color:{C_RED}; font-weight:700; font-size:{FS}px;")
            self._conv_dot.set_ok(False)
            self._lbl_clamp.setText("--")
            self._lbl_clamp.setStyleSheet(
                f"color:{C_GREY}; font-weight:700; font-size:{FS}px;")
            self._pill_conv.setText("UNREACHABLE")
            self._pill_conv.setStyleSheet(
                f"color:{C_RED}; font-size:12px; font-family:Rajdhani,'Arial',sans-serif;"
                f"font-weight:700; letter-spacing:1px;"
                f"border:1px solid rgba(255,61,85,0.35); border-radius:10px;"
                f"padding:1px 10px; background:rgba(255,61,85,0.06);")
            return

        mode = "NN + DLS" if self.controller.nn_network else "DLS only"
        cc  = C_NEON  if r["converged"] else C_RED
        clc = C_AMBER if r["clamped"]   else C_GREY

        self._lbl_mode.setText(mode)
        self._lbl_mode.setStyleSheet(f"color:{C_CYAN}; font-size:{FS}px;")
        self._lbl_iters.setText(str(r["iters"]))
        self._lbl_ms.setText(f"{r['solve_ms']:.1f} ms")
        self._lbl_conv.setText("✓ YES" if r["converged"] else "✗ NO")
        self._lbl_conv.setStyleSheet(
            f"color:{cc}; font-weight:700; font-size:{FS}px;")
        self._conv_dot.set_ok(r["converged"])
        self._lbl_clamp.setText("⚠ YES" if r["clamped"] else "—  NO")
        self._lbl_clamp.setStyleSheet(
            f"color:{clc}; font-weight:700; font-size:{FS}px;")

        # Header IK pill
        pill_col = "#39FF14" if r["converged"] else "#FF3D55"
        self._pill_conv.setStyleSheet(
            f"color:{pill_col}; font-size:12px; font-family:Rajdhani,'Arial',sans-serif;"
            f"font-weight:700; letter-spacing:1px;"
            f"border:1px solid rgba(57,255,20,0.30); border-radius:10px;"
            f"padding:1px 10px; background:rgba(57,255,20,0.06);")

    def _upd_status(self, r):
        held = r["held"]; snap = r.get("closest_snap", False)
        clamp = r["clamped"]

        if not r.get("reachable", True):
            txt = "◉  OUT OF RANGE  —  UNREACHABLE"
            col = C_RED
        elif held:
            txt = "◉  UNREACHABLE  — holding last valid"
            col = C_RED
        elif snap:
            txt = "◉  SNAPPED  — closest reachable pose"
            col = C_AMBER
        else:
            txt = "◉  REACHABLE"
            col = C_NEON
            self._last_ts = datetime.now().strftime("%H:%M:%S")

        self._badge_reach.setStyleSheet(
            f"color:{col}; font-size:13px; font-weight:700;")
        self._badge_reach.setText(txt)
        self._badge_clamp.setText("" if not r.get("reachable", True) else ("⚠  LIMIT CLAMP" if clamp else ""))
        self._badge_ts.setText(f"Last valid: {self._last_ts}")

    def _upd_ee(self, q_final, pos_m):
        T = forward_kinematics(q_final)
        p = T[:3, 3] * 1000.0
        self._lbl_ee_pos.setText(
            f"X= {p[0]:+7.1f}   Y= {p[1]:+7.1f}   Z= {p[2]:+7.1f}   mm")
        self._lbl_ee_tgt.setText(
            f"X= {pos_m[0]*1000:+7.1f}   "
            f"Y= {pos_m[1]*1000:+7.1f}   "
            f"Z= {pos_m[2]*1000:+7.1f}   mm")
        err = np.linalg.norm(p - pos_m * 1000.0)
        if err < 1.0:
            self._ee_match.setText("✓ IK SOLVED")
            self._ee_match.setStyleSheet(
                f"color:{C_NEON}; font-weight:700; font-size:{FS}px;")
        else:
            self._ee_match.setText("OUT OF RANGE")
            self._ee_match.setStyleSheet(
                f"color:{C_RED}; font-weight:700; font-size:{FS}px;")

    # ─────────────────────────────────────────────────────────────────────────
    #  BUTTONS
    # ─────────────────────────────────────────────────────────────────────────
    def _on_home(self):
        self.sl_x.set_value(300); self.sl_y.set_value(0)
        self.sl_z.set_value(350); self.sl_roll.set_value(0)
        self.sl_pitch.set_value(0); self.sl_yaw.set_value(0)
        self.controller.reset_to_home()

    def _on_demo_toggle(self, checked):
        self._demo_mode = checked; self._demo_t = 0.0

    def _on_random(self):
        self.sl_x.set_value(int(np.random.uniform(100, 600)))
        self.sl_y.set_value(int(np.random.uniform(-400, 400)))
        self.sl_z.set_value(int(np.random.uniform(150, 650)))

    def closeEvent(self, event):
        self._ctrl_tmr.stop(); self._rnd_tmr.stop()
        try: self.sim.close()
        except Exception: pass
        event.accept()
