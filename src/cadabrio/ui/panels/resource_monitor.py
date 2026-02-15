"""Resource Monitor panel for Cadabrio.

Floating window showing real-time system AND Cadabrio-specific resource usage:
CPU, GPU, RAM, VRAM, Disk I/O, and Network I/O.

Bars show system-wide usage as a dimmed background and Cadabrio's own
usage as a solid foreground overlay in the primary accent color.
Toggled from View menu (F9); updates every second.
"""

import os

import psutil
from loguru import logger

from PySide6.QtCore import Qt, QTimer, QRectF
from PySide6.QtGui import QColor, QPainter, QFont, QPen
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QGroupBox,
    QSizePolicy,
)


# --- Colors ---
_ACCENT = QColor("#39ff14")          # Neon green - Cadabrio process
_ACCENT_DIM = QColor(57, 255, 20, 60)  # Dimmed neon green - system total
_BAR_BG = QColor("#1a1a2e")          # Bar background
_BAR_BORDER = QColor("#2a2a45")
_TEXT_COLOR = QColor("#e0e0e0")
_TEXT_SHADOW = QColor(0, 0, 0, 140)


def _format_bytes(b: float) -> str:
    """Format bytes into a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


def _format_rate(b: float) -> str:
    """Format bytes/sec into a human-readable rate."""
    return f"{_format_bytes(b)}/s"


# ---------------------------------------------------------------------------
# Custom dual-layer bar widget
# ---------------------------------------------------------------------------

class DualBar(QWidget):
    """A progress bar showing system (dimmed) and process (solid) values.

    Both values are 0-100 percentages. The system bar renders behind
    the process bar so you can see Cadabrio's share of total usage.
    """

    BAR_HEIGHT = 22

    def __init__(self, parent=None):
        super().__init__(parent)
        self._system_pct = 0.0
        self._process_pct = 0.0
        self._label = ""
        self.setMinimumHeight(self.BAR_HEIGHT)
        self.setMaximumHeight(self.BAR_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_values(self, system_pct: float, process_pct: float, label: str = ""):
        """Update both values and repaint."""
        self._system_pct = max(0.0, min(100.0, system_pct))
        self._process_pct = max(0.0, min(100.0, process_pct))
        self._label = label
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        r = 3  # border radius

        # Background
        p.setPen(QPen(_BAR_BORDER, 1))
        p.setBrush(_BAR_BG)
        p.drawRoundedRect(0, 0, w - 1, h - 1, r, r)

        # System bar (dimmed)
        sys_w = int((w - 2) * self._system_pct / 100.0)
        if sys_w > 0:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_ACCENT_DIM)
            p.drawRoundedRect(1, 1, sys_w, h - 2, r, r)

        # Process bar (solid accent)
        proc_w = int((w - 2) * self._process_pct / 100.0)
        if proc_w > 0:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_ACCENT)
            p.drawRoundedRect(1, 1, proc_w, h - 2, r, r)

        # Label text
        if self._label:
            font = QFont("Segoe UI", 8)
            p.setFont(font)
            text_rect = QRectF(4, 0, w - 8, h)
            # Shadow
            p.setPen(_TEXT_SHADOW)
            p.drawText(text_rect.adjusted(1, 1, 1, 1), Qt.AlignmentFlag.AlignVCenter, self._label)
            # Text
            p.setPen(_TEXT_COLOR)
            p.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, self._label)

        p.end()


# ---------------------------------------------------------------------------
# Resource Monitor Window
# ---------------------------------------------------------------------------

class ResourceMonitorWindow(QWidget):
    """Floating resource monitor window."""

    UPDATE_INTERVAL_MS = 1000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cadabrio Resource Monitor")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.Tool)
        self.setMinimumSize(400, 600)
        self.resize(420, 660)

        # Track our own process
        self._process = psutil.Process(os.getpid())
        # Prime cpu_percent so first call returns meaningful data
        self._process.cpu_percent()

        self._prev_disk_io_sys = None
        self._prev_disk_io_proc = None
        self._prev_net_io = None

        # Callback for when window is closed via X button
        self.on_closed = None

        self._build_ui()

        # GPU monitoring via pynvml
        self._nvml_available = False
        self._gpu_handle = None
        self._our_pid = os.getpid()
        self._init_nvml()

        # Update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(self.UPDATE_INTERVAL_MS)

        # Initial read
        self._update()

    def _init_nvml(self):
        """Initialize NVIDIA Management Library for GPU monitoring."""
        try:
            import pynvml as nvml  # provided by nvidia-ml-py

            nvml.nvmlInit()
            self._gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
            name = nvml.nvmlDeviceGetName(self._gpu_handle)
            self._gpu_name_label.setText(name)
            self._nvml_available = True
            logger.debug(f"NVML initialized for GPU: {name}")
        except Exception as e:
            self._gpu_name_label.setText("GPU: Not available")
            logger.debug(f"NVML not available: {e}")

    # -------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Legend
        legend = QLabel(
            '<span style="color: #39ff14;">&#9632;</span> Cadabrio &nbsp;&nbsp;'
            '<span style="color: rgba(57,255,20,0.35);">&#9632;</span> System'
        )
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        legend.setStyleSheet("font-size: 9pt; padding: 2px;")
        layout.addWidget(legend)

        # --- CPU ---
        cpu_group = self._group("CPU")
        cpu_lay = QGridLayout(cpu_group)
        cpu_lay.setVerticalSpacing(6)
        self._cpu_bar = DualBar()
        self._cpu_freq_label = QLabel("")
        cpu_lay.addWidget(QLabel("Usage:"), 0, 0)
        cpu_lay.addWidget(self._cpu_bar, 0, 1)
        cpu_lay.addWidget(QLabel("Frequency:"), 1, 0)
        cpu_lay.addWidget(self._cpu_freq_label, 1, 1)
        layout.addWidget(cpu_group)

        # --- RAM ---
        ram_group = self._group("Memory")
        ram_lay = QGridLayout(ram_group)
        ram_lay.setVerticalSpacing(6)
        self._ram_bar = DualBar()
        self._ram_detail_label = QLabel("")
        ram_lay.addWidget(QLabel("Usage:"), 0, 0)
        ram_lay.addWidget(self._ram_bar, 0, 1)
        ram_lay.addWidget(QLabel("Detail:"), 1, 0)
        ram_lay.addWidget(self._ram_detail_label, 1, 1)
        layout.addWidget(ram_group)

        # --- GPU ---
        gpu_group = self._group("GPU")
        gpu_lay = QGridLayout(gpu_group)
        gpu_lay.setVerticalSpacing(10)
        gpu_lay.setContentsMargins(8, 12, 8, 8)

        self._gpu_name_label = QLabel("Detecting...")
        self._gpu_name_label.setStyleSheet("font-weight: bold; color: #39ff14;")
        gpu_lay.addWidget(self._gpu_name_label, 0, 0, 1, 2)

        gpu_lay.addWidget(QLabel("Utilization:"), 1, 0)
        self._gpu_util_bar = DualBar()
        gpu_lay.addWidget(self._gpu_util_bar, 1, 1)

        gpu_lay.addWidget(QLabel("VRAM:"), 2, 0)
        self._vram_bar = DualBar()
        gpu_lay.addWidget(self._vram_bar, 2, 1)

        gpu_lay.addWidget(QLabel("VRAM Detail:"), 3, 0)
        self._vram_detail_label = QLabel("")
        gpu_lay.addWidget(self._vram_detail_label, 3, 1)

        gpu_lay.addWidget(QLabel("Temperature:"), 4, 0)
        self._gpu_temp_label = QLabel("")
        gpu_lay.addWidget(self._gpu_temp_label, 4, 1)

        gpu_lay.addWidget(QLabel("Power:"), 5, 0)
        self._gpu_power_label = QLabel("")
        gpu_lay.addWidget(self._gpu_power_label, 5, 1)

        layout.addWidget(gpu_group)

        # --- Disk I/O ---
        disk_group = self._group("Disk I/O")
        disk_lay = QGridLayout(disk_group)
        disk_lay.setVerticalSpacing(6)
        disk_lay.addWidget(QLabel("Read:"), 0, 0)
        self._disk_read_label = QLabel("--")
        disk_lay.addWidget(self._disk_read_label, 0, 1)
        disk_lay.addWidget(QLabel("Write:"), 1, 0)
        self._disk_write_label = QLabel("--")
        disk_lay.addWidget(self._disk_write_label, 1, 1)
        layout.addWidget(disk_group)

        # --- Network I/O ---
        net_group = self._group("Network I/O")
        net_lay = QGridLayout(net_group)
        net_lay.setVerticalSpacing(6)
        net_lay.addWidget(QLabel("Recv:"), 0, 0)
        self._net_recv_label = QLabel("--")
        net_lay.addWidget(self._net_recv_label, 0, 1)
        net_lay.addWidget(QLabel("Sent:"), 1, 0)
        self._net_sent_label = QLabel("--")
        net_lay.addWidget(self._net_sent_label, 1, 1)
        layout.addWidget(net_group)

        layout.addStretch()

    def _group(self, title: str) -> QGroupBox:
        """Create a styled group box."""
        g = QGroupBox(title)
        return g

    # -------------------------------------------------------------------
    # Update methods
    # -------------------------------------------------------------------

    def _update(self):
        self._update_cpu()
        self._update_ram()
        self._update_gpu()
        self._update_disk()
        self._update_network()

    def _update_cpu(self):
        sys_pct = psutil.cpu_percent(interval=None)
        num_cpus = psutil.cpu_count() or 1
        try:
            proc_pct = self._process.cpu_percent() / num_cpus
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_pct = 0.0

        self._cpu_bar.set_values(
            sys_pct, proc_pct,
            f"Cadabrio {proc_pct:.1f}%  |  System {sys_pct:.1f}%"
        )

        freq = psutil.cpu_freq()
        if freq:
            self._cpu_freq_label.setText(f"{freq.current:.0f} MHz")

    def _update_ram(self):
        mem = psutil.virtual_memory()
        sys_pct = mem.percent

        try:
            proc_mem = self._process.memory_info()
            proc_bytes = proc_mem.rss
            proc_pct = (proc_bytes / mem.total) * 100 if mem.total else 0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            proc_bytes = 0
            proc_pct = 0.0

        self._ram_bar.set_values(
            sys_pct, proc_pct,
            f"Cadabrio {_format_bytes(proc_bytes)}  |  System {sys_pct:.1f}%"
        )
        self._ram_detail_label.setText(
            f"{_format_bytes(mem.used)} / {_format_bytes(mem.total)}  "
            f"({_format_bytes(mem.available)} free)"
        )

    def _update_gpu(self):
        if not self._nvml_available:
            return

        try:
            import pynvml as nvml  # provided by nvidia-ml-py

            # System GPU utilization
            util = nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            sys_gpu_pct = util.gpu

            # System VRAM
            mem_info = nvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            sys_vram_pct = (mem_info.used / mem_info.total * 100) if mem_info.total else 0

            # Per-process VRAM (Cadabrio's share)
            proc_vram_bytes = 0
            try:
                procs = nvml.nvmlDeviceGetComputeRunningProcesses(self._gpu_handle)
                for p in procs:
                    if p.pid == self._our_pid:
                        proc_vram_bytes = p.usedGpuMemory or 0
                        break
            except Exception:
                pass
            # Also check graphics processes
            try:
                gprocs = nvml.nvmlDeviceGetGraphicsRunningProcesses(self._gpu_handle)
                for p in gprocs:
                    if p.pid == self._our_pid:
                        proc_vram_bytes = max(proc_vram_bytes, p.usedGpuMemory or 0)
                        break
            except Exception:
                pass

            proc_vram_pct = (proc_vram_bytes / mem_info.total * 100) if mem_info.total else 0

            self._gpu_util_bar.set_values(
                sys_gpu_pct, 0,  # Per-process GPU util not available via NVML
                f"System {sys_gpu_pct}%"
            )

            self._vram_bar.set_values(
                sys_vram_pct, proc_vram_pct,
                f"Cadabrio {_format_bytes(proc_vram_bytes)}  |  "
                f"System {sys_vram_pct:.1f}%"
            )
            self._vram_detail_label.setText(
                f"{_format_bytes(mem_info.used)} / {_format_bytes(mem_info.total)}"
            )

            # Temperature
            try:
                temp = nvml.nvmlDeviceGetTemperature(
                    self._gpu_handle, nvml.NVML_TEMPERATURE_GPU
                )
                self._gpu_temp_label.setText(f"{temp} °C")
            except Exception:
                self._gpu_temp_label.setText("N/A")

            # Power
            try:
                power_mw = nvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                power_limit_mw = nvml.nvmlDeviceGetPowerManagementLimit(self._gpu_handle)
                self._gpu_power_label.setText(
                    f"{power_mw / 1000:.0f} W / {power_limit_mw / 1000:.0f} W"
                )
            except Exception:
                self._gpu_power_label.setText("N/A")

        except Exception as e:
            logger.debug(f"GPU update error: {e}")

    def _update_disk(self):
        # System disk I/O
        disk_sys = psutil.disk_io_counters()
        if disk_sys is None:
            return

        # Process disk I/O
        try:
            disk_proc = self._process.io_counters()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            disk_proc = None

        if self._prev_disk_io_sys is not None:
            sys_read = disk_sys.read_bytes - self._prev_disk_io_sys.read_bytes
            sys_write = disk_sys.write_bytes - self._prev_disk_io_sys.write_bytes
            proc_read = 0
            proc_write = 0
            if disk_proc and self._prev_disk_io_proc:
                proc_read = disk_proc.read_bytes - self._prev_disk_io_proc.read_bytes
                proc_write = disk_proc.write_bytes - self._prev_disk_io_proc.write_bytes

            self._disk_read_label.setText(
                f"Cadabrio {_format_rate(proc_read)}  |  System {_format_rate(sys_read)}"
            )
            self._disk_write_label.setText(
                f"Cadabrio {_format_rate(proc_write)}  |  System {_format_rate(sys_write)}"
            )

        self._prev_disk_io_sys = disk_sys
        self._prev_disk_io_proc = disk_proc

    def _update_network(self):
        net = psutil.net_io_counters()
        if net is None:
            return

        if self._prev_net_io is not None:
            recv_rate = net.bytes_recv - self._prev_net_io.bytes_recv
            sent_rate = net.bytes_sent - self._prev_net_io.bytes_sent
            self._net_recv_label.setText(f"System {_format_rate(recv_rate)}")
            self._net_sent_label.setText(f"System {_format_rate(sent_rate)}")

        self._prev_net_io = net

    # -------------------------------------------------------------------
    # Window lifecycle
    # -------------------------------------------------------------------

    def closeEvent(self, event):
        self._timer.stop()
        if self.on_closed:
            self.on_closed()
        event.accept()

    def showEvent(self, event):
        if not self._timer.isActive():
            self._timer.start(self.UPDATE_INTERVAL_MS)
        super().showEvent(event)
