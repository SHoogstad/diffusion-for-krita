"""Krita's DoubleSliderSpinBox Python wrapper, with a fallback for Krita 5 builds that don't
expose it. The fallback mimics Krita's slider spinbox: a number box with a fill bar that can be
dragged to change the value. Double-click to type a value."""

try:
    from krita import DoubleSliderSpinBox
except ImportError:
    from ..qt.QtCore import QEvent, QObject, QRectF, Qt, pyqtSignal
    from ..qt.QtGui import QColor, QPainter, QPalette
    from ..qt.QtWidgets import QDoubleSpinBox

    class _SliderSpinBox(QDoubleSpinBox):
        drag_finished = pyqtSignal()

        def __init__(self):
            super().__init__()
            self.soft_min: float | None = None
            self.soft_max: float | None = None
            self.dragging = False
            self._pressed = False
            self._editing = False
            self.setKeyboardTracking(False)
            # Transparent text field so the fill bar painted by the spinbox shows behind the text
            palette = self.lineEdit().palette()
            palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0, 0))
            self.lineEdit().setPalette(palette)
            self.lineEdit().setReadOnly(True)
            self.lineEdit().installEventFilter(self)
            self.lineEdit().editingFinished.connect(self._stop_editing)

        def _bar_range(self):
            lo = self.minimum() if self.soft_min is None else self.soft_min
            hi = self.maximum() if self.soft_max is None else self.soft_max
            return lo, max(hi, lo + 1e-9)

        def _value_at(self, x: float):
            lo, hi = self._bar_range()
            width = max(self.lineEdit().width(), 1)
            value = lo + (hi - lo) * min(max(x / width, 0.0), 1.0)
            return round(value, self.decimals())

        def _stop_editing(self):
            self._editing = False
            self.lineEdit().setReadOnly(True)
            self.lineEdit().deselect()
            self.update()

        def eventFilter(self, obj, event):
            if obj is not self.lineEdit() or self._editing or not self.isEnabled():
                return super().eventFilter(obj, event)
            kind = event.type()
            if kind == QEvent.Type.MouseButtonDblClick:
                self._editing = True
                self._pressed = self.dragging = False
                self.lineEdit().setReadOnly(False)
                self.lineEdit().setFocus()
                self.selectAll()
                return True
            if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._pressed = True
                self.dragging = True
                self.setValue(self._value_at(event.pos().x()))
                return True
            if kind == QEvent.Type.MouseMove and self._pressed:
                self.setValue(self._value_at(event.pos().x()))
                return True
            if kind == QEvent.Type.MouseButtonRelease and self._pressed:
                self._pressed = False
                self.dragging = False
                self.drag_finished.emit()
                return True
            return super().eventFilter(obj, event)

        def stepBy(self, steps: int):
            super().stepBy(steps)
            if not self._editing:
                self.lineEdit().deselect()

        def focusOutEvent(self, event):
            if self._editing:
                self._stop_editing()
            super().focusOutEvent(event)

        def paintEvent(self, event):
            super().paintEvent(event)
            lo, hi = self._bar_range()
            ratio = min(max((self.value() - lo) / (hi - lo), 0.0), 1.0)
            if ratio <= 0:
                return
            rect = QRectF(self.lineEdit().geometry()).adjusted(-1, -1, 1, 1)
            rect.setWidth(rect.width() * ratio)
            color = QColor(self.palette().color(QPalette.ColorRole.Highlight))
            color.setAlpha(110 if self.isEnabled() else 45)
            painter = QPainter(self)
            painter.fillRect(rect, color)

    class DoubleSliderSpinBox(QObject):  # type: ignore[no-redef]
        draggingFinished = pyqtSignal()

        def __init__(self):
            super().__init__()
            self._widget = _SliderSpinBox()
            self._widget.drag_finished.connect(self.draggingFinished)

        def widget(self):
            return self._widget

        def stepBy(self, steps: int):
            self._widget.stepBy(steps)

        def setRange(self, min: float, max: float, decimals: int = 0, compute_fast_step=True):
            self._widget.setDecimals(decimals)
            self._widget.setRange(min, max)

        def setValue(self, value: float):
            self._widget.setValue(value)

        def value(self):
            return self._widget.value()

        def setSoftMinimum(self, value: float):
            self._widget.soft_min = value
            self._widget.update()

        def setSoftMaximum(self, value: float):
            self._widget.soft_max = value
            self._widget.update()

        def isDragging(self) -> bool:
            return self._widget.dragging
