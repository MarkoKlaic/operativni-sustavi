import sys
import time
import math
from PyQt5 import QtWidgets, QtCore, QtGui
from tsp import (
    distance_matrix,
    held_karp,
    nearest_neighbor,
    two_opt,
    simulated_annealing,
    tour_length,
)
from PyQt5.QtSvg import QSvgGenerator


class SolverThread(QtCore.QThread):
    resultReady = QtCore.pyqtSignal(float, list, str, float)

    def __init__(self, points, mode="heur"):
        super().__init__()
        self.points = points
        self.mode = mode

    def run(self):
        if not self.points:
            return
        dist = distance_matrix(self.points)
        t0 = time.time()
        if self.mode == "exact":
            cost, tour = held_karp(dist)
            name = "Held-Karp"
        elif self.mode == "heur":
            tour = nearest_neighbor(dist)
            tour = two_opt(tour, dist)
            cost = tour_length(tour, dist)
            name = "NN+2-opt"
        else:
            tour = simulated_annealing(dist)
            cost = tour_length(tour, dist)
            name = "Simulated Annealing"
        t1 = time.time()
        try:
            self.resultReady.emit(cost, tour, name, t1 - t0)
        except Exception:
            pass


class Canvas(QtWidgets.QWidget):
    pointsChanged = QtCore.pyqtSignal()
    animationUpdated = QtCore.pyqtSignal(int, float)
    animationFinished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.tour = []
        self.setMinimumSize(600, 400)
        self._anim_tour = None
        self._anim_idx = 0
        self._anim_pos = 0.0
        self._anim_delta = 0.25
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.timeout.connect(self._anim_step)

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = self.rect()
        grad = QtGui.QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QtGui.QColor(245, 245, 250))
        grad.setColorAt(1.0, QtGui.QColor(225, 230, 240))
        qp.fillRect(rect, QtGui.QBrush(grad))

        qp.setPen(QtGui.QPen(QtGui.QColor(240, 240, 245), 1))
        step = 40
        for x in range(0, rect.width(), step):
            qp.drawLine(x, 0, x, rect.height())
        for y in range(0, rect.height(), step):
            qp.drawLine(0, y, rect.width(), y)

        active_tour = None
        animating = self._anim_tour is not None
        if animating:
            active_tour = self._anim_tour[: int(self._anim_pos) + 1]
        else:
            active_tour = self.tour

        if active_tour and len(active_tour) > 0:
            seq = []
            if animating:
                pos = self._anim_pos
                idx = int(pos)
                frac = pos - idx
                for k in range(min(len(self._anim_tour), idx + 1)):
                    ii = self._anim_tour[k]
                    if 0 <= ii < len(self.points):
                        seq.append(self.points[ii])
                if idx + 1 < len(self._anim_tour):
                    ai = self._anim_tour[idx]
                    bi = self._anim_tour[idx + 1]
                    if 0 <= ai < len(self.points) and 0 <= bi < len(self.points):
                        a = self.points[ai]
                        b = self.points[bi]
                        inter_x = a[0] + (b[0] - a[0]) * frac
                        inter_y = a[1] + (b[1] - a[1]) * frac
                        seq.append((inter_x, inter_y))
            else:
                seq = [self.points[i] for i in active_tour if 0 <= i < len(self.points)]

            if seq:
                pen = QtGui.QPen(QtGui.QColor(80, 160, 255), 3)
                pen.setCapStyle(QtCore.Qt.RoundCap)
                qp.setPen(pen)
                if animating:
                    poly = QtGui.QPolygonF()
                    for xx, yy in seq:
                        poly.append(QtCore.QPointF(xx, yy))
                    qp.drawPolyline(poly)
                else:
                    path = self._catmull_rom_path(seq)
                    qp.drawPath(path)

            if animating:
                pos = self._anim_pos
                idx = int(pos)
                frac = pos - idx
                if idx + 1 < len(self._anim_tour):
                    ai = self._anim_tour[idx]
                    bi = self._anim_tour[idx + 1]
                    if 0 <= ai < len(self.points) and 0 <= bi < len(self.points):
                        a = self.points[ai]
                        b = self.points[bi]
                    else:
                        a = b = None
                else:
                    a = b = None
                if a is not None and b is not None:
                    x = a[0] + (b[0] - a[0]) * frac
                    y = a[1] + (b[1] - a[1]) * frac
                    glow = QtGui.QRadialGradient(x, y, 16)
                    glow.setColorAt(0.0, QtGui.QColor(255, 210, 120, 240))
                    glow.setColorAt(0.6, QtGui.QColor(255, 150, 60, 160))
                    glow.setColorAt(1.0, QtGui.QColor(255, 150, 60, 0))
                    qp.setBrush(QtGui.QBrush(glow))
                    qp.setPen(QtCore.Qt.NoPen)
                    qp.drawEllipse(int(x) - 16, int(y) - 16, 32, 32)
                    qp.setBrush(QtGui.QBrush(QtGui.QColor(255, 200, 80)))
                    qp.setPen(QtGui.QPen(QtCore.Qt.black, 1))
                    qp.drawEllipse(int(x) - 6, int(y) - 6, 12, 12)

        start_idx = None
        end_idx = None
        if animating:
            if self._anim_tour:
                start_idx = self._anim_tour[0]
                end_idx = self._anim_tour[
                    min(int(self._anim_pos), len(self._anim_tour) - 1)
                ]
        else:
            if self.tour:
                start_idx = self.tour[0]
                end_idx = self.tour[-1]

        for i, p in enumerate(self.points):
            is_start = i == start_idx
            is_end = i == end_idx
            qp.setPen(QtCore.Qt.black)
            if is_start:
                qp.setBrush(QtGui.QBrush(QtGui.QColor(50, 205, 50)))
            elif is_end:
                qp.setBrush(QtGui.QBrush(QtGui.QColor(220, 20, 60)))
            else:
                qp.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
            size = 10
            if is_start or is_end:
                size = 12
            qp.drawEllipse(int(p[0]) - size // 2, int(p[1]) - size // 2, size, size)
            qp.drawText(int(p[0]) + 8, int(p[1]) + 8, str(i))
            if is_start:
                qp.drawText(int(p[0]) - 6, int(p[1]) - 8, "S")
            if is_end:
                qp.drawText(int(p[0]) - 6, int(p[1]) - 8, "E")

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.points.append((event.x(), event.y()))
            self.tour = []
            self.update()
            try:
                self.pointsChanged.emit()
            except Exception:
                pass

    def clear(self):
        try:
            if hasattr(self, "_anim_timer") and self._anim_timer.isActive():
                self._anim_timer.stop()
        except Exception:
            pass
        self._anim_tour = None
        self._anim_idx = 0
        self._anim_pos = 0.0
        self.points = []
        self.tour = []
        self.update()
        try:
            self.pointsChanged.emit()
        except Exception:
            pass

    def set_tour(self, tour):
        self.animate_tour(
            tour,
            interval_ms=(
                MainWindow.DEFAULT_SPEED_MS
                if hasattr(MainWindow, "DEFAULT_SPEED_MS")
                else 300
            ),
        )

    def _catmull_rom_path(self, pts):
        path = QtGui.QPainterPath()
        if not pts:
            return path
        n = len(pts)
        if n == 1:
            path.moveTo(pts[0][0], pts[0][1])
            return path

        def p(i):
            if i < 0:
                return pts[0]
            if i >= n:
                return pts[-1]
            return pts[i]

        path.moveTo(pts[0][0], pts[0][1])
        for i in range(n - 1):
            P0 = p(i - 1)
            P1 = p(i)
            P2 = p(i + 1)
            P3 = p(i + 2)
            B1x = P1[0] + (P2[0] - P0[0]) / 6.0
            B1y = P1[1] + (P2[1] - P0[1]) / 6.0
            B2x = P2[0] - (P3[0] - P1[0]) / 6.0
            B2y = P2[1] - (P3[1] - P1[1]) / 6.0
            path.cubicTo(B1x, B1y, B2x, B2y, P2[0], P2[1])
        return path

    def animate_tour(self, tour, interval_ms=150):
        if not tour:
            return
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        self._anim_tour = list(tour)
        self._anim_idx = 0
        self._anim_pos = 0.0
        self._anim_delta = max(0.03, 0.25 * (150.0 / float(max(1, interval_ms))))
        self._is_paused = False
        self._anim_timer.start(interval_ms)

    def pause_animation(self):
        if self._anim_timer.isActive():
            self._anim_timer.stop()
            self._is_paused = True

    def resume_animation(self, interval_ms=150):
        if self._anim_tour is None:
            return
        if not self._anim_timer.isActive():
            self._is_paused = False
            self._anim_delta = max(0.03, 0.25 * (150.0 / float(max(1, interval_ms))))
            self._anim_timer.start(interval_ms)

    def step_animation(self):
        if self._anim_tour is None:
            return
        if self._anim_timer.isActive():
            self._anim_timer.stop()
        self._anim_pos += self._anim_delta
        if self._anim_pos >= len(self._anim_tour) - 1:
            self._anim_timer.stop()
            self.tour = list(self._anim_tour)
            self._anim_tour = None
            self._anim_pos = 0.0
            self._anim_idx = 0
            self.update()
            self.animationFinished.emit()
            return
        self._anim_idx = int(self._anim_pos)
        self.update()
        length = self._traveled_length()
        self.animationUpdated.emit(self._anim_idx, length)

    def _traveled_length(self):
        if not self._anim_tour:
            return 0.0
        pos = self._anim_pos
        idx = int(pos)
        frac = pos - idx
        s = 0.0
        for k in range(idx):
            a = self.points[self._anim_tour[k]]
            b = self.points[self._anim_tour[k + 1]]
            s += math.hypot(a[0] - b[0], a[1] - b[1])
        if idx + 1 < len(self._anim_tour):
            a = self.points[self._anim_tour[idx]]
            b = self.points[self._anim_tour[idx + 1]]
            s += math.hypot(a[0] - b[0], a[1] - b[1]) * frac
        return s

    def _anim_step(self):
        if self._anim_tour is None:
            return
        self._anim_pos += self._anim_delta
        if self._anim_pos >= len(self._anim_tour) - 1:
            self._anim_timer.stop()
            self.tour = list(self._anim_tour)
            self._anim_tour = None
            self._anim_pos = 0.0
            self._anim_idx = 0
            self.update()
            self.animationFinished.emit()
            return
        self._anim_idx = int(self._anim_pos)
        length = self._traveled_length()
        self.update()
        try:
            self.animationUpdated.emit(self._anim_idx, length)
        except Exception:
            pass


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TSP Solver — Exact & Heuristics")
        self.canvas = Canvas()
        MainWindow.DEFAULT_SPEED_MS = 300

        btn_clear = QtWidgets.QPushButton("Clear")
        btn_random = QtWidgets.QPushButton("Random")
        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(2, 200)
        self.spin_n.setValue(10)
        btn_exact = QtWidgets.QPushButton("Solve Exact (Held-Karp)")
        btn_heur = QtWidgets.QPushButton("Heuristic (NN+2-opt)")
        btn_sa = QtWidgets.QPushButton("Simulated Annealing")

        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setRange(50, 1000)
        self.speed_slider.setValue(MainWindow.DEFAULT_SPEED_MS)
        self.speed_label = QtWidgets.QLabel(
            f"Anim speed: {self.speed_slider.value()} ms"
        )

        self.coords_list = QtWidgets.QListWidget()
        self.info_display = QtWidgets.QTextEdit()
        self.info_display.setReadOnly(True)
        self.info_display.setFixedWidth(240)

        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(self.canvas)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(btn_clear)
        controls_layout.addWidget(btn_random)
        controls_layout.addWidget(QtWidgets.QLabel("Random N:"))
        controls_layout.addWidget(self.spin_n)
        left_layout.addLayout(controls_layout)

        algo_layout = QtWidgets.QHBoxLayout()
        algo_layout.addWidget(btn_exact)
        algo_layout.addWidget(btn_heur)
        algo_layout.addWidget(btn_sa)

        btn_pause = QtWidgets.QPushButton("Pause")
        btn_step = QtWidgets.QPushButton("Step")
        algo_layout.addWidget(btn_pause)
        algo_layout.addWidget(btn_step)
        left_layout.addLayout(algo_layout)

        speed_layout = QtWidgets.QHBoxLayout()
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_slider)
        left_layout.addLayout(speed_layout)

        main_layout = QtWidgets.QHBoxLayout()
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_layout)
        main_layout.addWidget(left_widget, stretch=1)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(QtWidgets.QLabel("Coordinates"))
        right_layout.addWidget(self.coords_list)
        right_layout.addWidget(QtWidgets.QLabel("Solution info"))
        right_layout.addWidget(self.info_display)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget)

        container = QtWidgets.QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        btn_clear.clicked.connect(self.on_clear)
        btn_random.clicked.connect(self.on_random)
        btn_exact.clicked.connect(self.on_exact)
        btn_heur.clicked.connect(self.on_heur)
        btn_sa.clicked.connect(self.on_sa)
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_label.setText(f"Anim speed: {v} ms")
        )
        btn_pause.clicked.connect(self.on_pause_toggle)
        btn_step.clicked.connect(self.on_step)

        self.canvas.pointsChanged.connect(self.update_coords)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_info)
        self.timer.start(300)

        self.canvas.animationUpdated.connect(self.on_anim_update)
        self.canvas.animationFinished.connect(self.on_anim_finished)
        self._last_tour = None

        self.setStyleSheet(
            """
        QMainWindow { background: #2b2b2b; }
        QWidget { color: #dddddd; background: transparent; }
        QPushButton {
            padding: 6px;
            background: #3c3f41;
            border: 1px solid #555;
            border-radius: 4px;
        }
        QPushButton:hover { background: #4c5052; }
        QSlider::groove:horizontal { height: 8px; background: #3a3a3a; }
        QSlider::handle:horizontal { background: #777; width: 12px; }
        QListWidget {
            font-family: Consolas, monospace;
            background: #262626;
            border: 1px solid #444;
        }
        QTextEdit { background: #1e1e1e; border: 1px solid #444; }
        QToolBar { background: #333333; spacing: 6px }
        """
        )

        toolbar = QtWidgets.QToolBar("Main")
        self.addToolBar(toolbar)
        act_clear = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon), "Clear", self
        )
        act_random = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload), "Random", self
        )
        act_save = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton),
            "Save PNG",
            self,
        )
        act_save_svg = QtWidgets.QAction(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton),
            "Save SVG",
            self,
        )
        act_exact = QtWidgets.QAction("Exact", self)
        act_heur = QtWidgets.QAction("Heuristic", self)
        act_sa = QtWidgets.QAction("SA", self)
        toolbar.addAction(act_clear)
        toolbar.addAction(act_random)
        toolbar.addSeparator()
        toolbar.addAction(act_exact)
        toolbar.addAction(act_heur)
        toolbar.addAction(act_sa)
        toolbar.addSeparator()
        toolbar.addAction(act_save)
        toolbar.addAction(act_save_svg)

        act_clear.triggered.connect(self.on_clear)
        act_random.triggered.connect(self.on_random)
        act_save.triggered.connect(self.save_png)
        act_save_svg.triggered.connect(self.save_svg)
        act_exact.triggered.connect(lambda: self.start_solver("exact"))
        act_heur.triggered.connect(lambda: self.start_solver("heur"))
        act_sa.triggered.connect(lambda: self.start_solver("sa"))

    def _update_info(self):
        self.statusBar().showMessage(f"Points: {len(self.canvas.points)}")

    def on_clear(self):
        self.canvas.clear()
        self.update_coords()

    def on_random(self):
        n = self.spin_n.value()
        w = self.canvas.width()
        h = self.canvas.height()
        import random

        self.canvas.points = [
            (random.randint(20, w - 20), random.randint(20, h - 20)) for _ in range(n)
        ]
        self.canvas.tour = []
        self.canvas.update()
        self.update_coords()

    def on_exact(self):
        self.start_solver("exact")

    def on_heur(self):
        self.start_solver("heur")

    def on_sa(self):
        self.start_solver("sa")

    def start_solver(self, mode):
        if not self.canvas.points:
            return
        if mode == "exact" and len(self.canvas.points) > 14:
            self.info_display.setPlainText(
                "Too many points for exact solver (limit ~14)."
            )
            return
        self._set_controls_enabled(False)
        self.info_display.setPlainText(f"Running {mode}...")
        self.worker = SolverThread(list(self.canvas.points), mode=mode)
        self.worker.resultReady.connect(self._on_solver_result)
        self.worker.start()

    def _on_solver_result(self, cost, tour, name, duration):
        self.info_static = (
            f"Algorithm: {name}\n"
            f"Length: {cost:.3f}\n"
            f"Time: {duration:.3f}s\n"
            f"Cities: {len(self.canvas.points)}"
        )
        self.info_display.setPlainText(self.info_static)
        self._last_tour = list(tour) if tour is not None else None
        QtCore.QTimer.singleShot(
            50,
            lambda: self.canvas.animate_tour(
                tour, interval_ms=self.speed_slider.value()
            ),
        )
        self._set_controls_enabled(True)

    def _set_controls_enabled(self, enabled: bool):
        widgets = [self.spin_n, self.speed_slider, self.coords_list]
        for w in widgets:
            try:
                w.setEnabled(enabled)
            except Exception:
                pass

    def save_png(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save PNG", "", "PNG Files (*.png)"
        )
        if not fn:
            return
        rect = self.canvas.rect()
        img = QtGui.QImage(rect.size(), QtGui.QImage.Format_ARGB32)
        img.fill(QtGui.QColor(0, 0, 0, 0))
        painter = QtGui.QPainter(img)
        self.canvas.render(painter)
        painter.end()
        img.save(fn)

    def save_svg(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save SVG", "", "SVG Files (*.svg)"
        )
        if not fn:
            return
        gen = QSvgGenerator()
        gen.setFileName(fn)
        size = self.canvas.size()
        gen.setSize(size)
        gen.setViewBox(self.canvas.rect())
        gen.setTitle("TSP tour")
        painter = QtGui.QPainter(gen)
        self.canvas.render(painter)
        painter.end()

    def update_coords(self):
        self.coords_list.clear()
        for i, p in enumerate(self.canvas.points):
            self.coords_list.addItem(f"{i}: ({int(p[0])}, {int(p[1])})")

    def on_pause_toggle(self):
        if getattr(self, "_paused", False):
            self.canvas.resume_animation(interval_ms=self.speed_slider.value())
            self._paused = False
        else:
            self.canvas.pause_animation()
            self._paused = True

    def on_step(self):
        self.canvas.step_animation()

    def on_anim_update(self, idx, length):
        total = len(self.canvas._anim_tour) if self.canvas._anim_tour else 0
        static = getattr(self, "info_static", "")
        dyn = f"Animating: step {idx}/{total}  Traveled: {length:.3f}"
        if static:
            self.info_display.setPlainText(static + "\n" + dyn)
        else:
            self.info_display.setPlainText(dyn)

    def on_anim_finished(self):
        static = getattr(self, "info_static", "")
        if static:
            final_len = tour_length(
                self.canvas.tour, distance_matrix(self.canvas.points)
            )
            txt = static + f"\nFinal traveled: {final_len:.3f}"
            tour_seq = (
                self._last_tour
                if getattr(self, "_last_tour", None)
                else getattr(self.canvas, "tour", None)
            )
            if tour_seq:
                try:
                    seq_txt = " -> ".join(str(int(i)) for i in tour_seq)
                    txt += "\nTour order: " + seq_txt
                except Exception:
                    pass
            self.info_display.setPlainText(txt)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
