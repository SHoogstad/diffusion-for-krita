"""Krita's DoubleSliderSpinBox Python wrapper, with a fallback for Krita versions before 5.3
that don't expose it. The fallback is a plain QDoubleSpinBox without the slider bar."""

try:
    from krita import DoubleSliderSpinBox
except ImportError:
    from PyQt5.QtCore import QObject, pyqtSignal
    from PyQt5.QtWidgets import QDoubleSpinBox

    class DoubleSliderSpinBox(QObject):  # type: ignore[no-redef]
        draggingFinished = pyqtSignal()

        def __init__(self):
            super().__init__()
            self._widget = QDoubleSpinBox()
            self._widget.setKeyboardTracking(False)

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
            pass

        def setSoftMaximum(self, value: float):
            pass

        def isDragging(self) -> bool:
            return False
