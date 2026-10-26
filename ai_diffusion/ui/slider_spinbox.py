"""Krita's DoubleSliderSpinBox Python wrapper, with a fallback for Krita builds where it can't
be used. The fallback mimics Krita's slider spinbox: a number box with a fill bar that can be
dragged to change the value. Double-click to type a value.

Callers use widget() as the QDoubleSpinBox (prefix, suffix, valueChanged) and put it in their
layouts. Krita's slider is a QDoubleSpinBox subclass, but some builds hand it to Python typed as a
plain QWidget; it is cast back. If widget() is not a spinbox at all, the fallback is used.
"""

from typing import TYPE_CHECKING

from ..qt import sip
from ..qt.QtCore import QEvent, QObject, QRectF, Qt, pyqtSignal
from ..qt.QtGui import QColor, QMouseEvent, QPainter, QPalette
from ..qt.QtWidgets import QApplication, QDoubleSpinBox, QLineEdit, QWidget
from ..util import client_logger as log
from ..util import ensure


def spin_box(widget: QWidget | None) -> QDoubleSpinBox | None:
    if widget is None:
        return None
    if isinstance(widget, QDoubleSpinBox):
        return widget
    if widget.inherits("QDoubleSpinBox"):  # Krita subclass Python only knows as QWidget
        return sip.cast(widget, QDoubleSpinBox)
    return None


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
        palette = self._line_edit().palette()
        palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0, 0))
        self._line_edit().setPalette(palette)
        self._line_edit().setReadOnly(True)
        self._line_edit().installEventFilter(self)
        self._line_edit().editingFinished.connect(self._stop_editing)

    def _line_edit(self) -> QLineEdit:
        return ensure(self.lineEdit())

    def _bar_range(self):
        lo = self.minimum() if self.soft_min is None else self.soft_min
        hi = self.maximum() if self.soft_max is None else self.soft_max
        return lo, max(hi, lo + 1e-9)

    def _value_at(self, x: float):
        lo, hi = self._bar_range()
        width = max(self._line_edit().width(), 1)
        value = lo + (hi - lo) * min(max(x / width, 0.0), 1.0)
        return round(value, self.decimals())

    def _stop_editing(self):
        self._editing = False
        self._line_edit().setReadOnly(True)
        self._line_edit().deselect()
        self.update()

    def eventFilter(self, a0, a1):
        event = a1
        if a0 is not self._line_edit() or not isinstance(event, QMouseEvent):
            return super().eventFilter(a0, a1)
        if self._editing or not self.isEnabled():
            return super().eventFilter(a0, a1)
        kind = event.type()
        if kind == QEvent.Type.MouseButtonDblClick:
            self._editing = True
            self._pressed = self.dragging = False
            self._line_edit().setReadOnly(False)
            self._line_edit().setFocus()
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
        return super().eventFilter(a0, a1)

    def stepBy(self, steps: int):
        super().stepBy(steps)
        if not self._editing:
            self._line_edit().deselect()

    def focusOutEvent(self, e):
        if self._editing:
            self._stop_editing()
        super().focusOutEvent(e)

    def paintEvent(self, e):
        super().paintEvent(e)
        lo, hi = self._bar_range()
        ratio = min(max((self.value() - lo) / (hi - lo), 0.0), 1.0)
        if ratio <= 0:
            return
        rect = QRectF(self._line_edit().geometry()).adjusted(-1, -1, 1, 1)
        rect.setWidth(rect.width() * ratio)
        color = QColor(self.palette().color(QPalette.ColorRole.Highlight))
        color.setAlpha(110 if self.isEnabled() else 45)
        painter = QPainter(self)
        painter.fillRect(rect, color)


class FallbackDoubleSliderSpinBox(QObject):
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


def _native_slider():
    try:
        from krita import DoubleSliderSpinBox as Native
    except ImportError:
        return None

    class NativeDoubleSliderSpinBox(Native):
        def widget(self):  # type: ignore (returns the spinbox inside)
            return ensure(spin_box(super().widget()))

    if isinstance(QApplication.instance(), QApplication):  # widgets need a QApplication
        try:
            probe = Native()
            widget = probe.widget()
            found = spin_box(widget)
            log.info(f"Krita DoubleSliderSpinBox widget: {type(widget).__name__}, spinbox: {found}")
            if widget is not None:
                widget.deleteLater()
            if found is None:
                return None
        except Exception as e:
            log.warning(f"Krita DoubleSliderSpinBox can't be used: {e}")
            return None
    return NativeDoubleSliderSpinBox


if TYPE_CHECKING:
    from krita import DoubleSliderSpinBox
else:
    DoubleSliderSpinBox = _native_slider() or FallbackDoubleSliderSpinBox
