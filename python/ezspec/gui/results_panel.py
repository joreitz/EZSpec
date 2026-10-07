"""Fit statistics, warnings, derived quantities, correlations and residual diagnostics."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..fit import stats as S
from ..fit.result import fmt_value
from .theme import SEVERITY_COLOR, SEVERITY_ICON

TIPS = {
    "σ source": "Where the uncertainties come from. χ² follows a χ² distribution only if σ is known or "
                "properly estimated.",
    "Covariance": "absolute: σ is taken as known. scaled: standard errors scaled by √χ²_ν (or s) – "
                  "required when σ is unknown.",
    "χ²_ν": "Reduced chi-square χ²/ν. For a correct model with correct σ, expected ≈ 1 ± √(2/ν).",
    "s": "Residual standard deviation √(RSS/ν) – estimate of the noise when σ is unknown.",
    "AICc": "Small-sample corrected Akaike information criterion (smaller = better). Only differences between "
            "models fitted to identical data are meaningful.",
    "R²": "Descriptive only. Not a suitable criterion for comparing nonlinear models (Spiess & Neumeyer 2010).",
    "Runs test": "Wald–Wolfowitz runs test on the residual signs: too few sign changes indicate systematic "
                 "misfit.",
    "Condition number": "Ratio of the largest to the smallest singular value of the weighted Jacobian; "
                        "> 1e12: parameters practically not identifiable.",
}


def _g(v, spec=".5g"):
    if v is None:
        return "–"
    try:
        return format(v, spec) if np.isfinite(v) else str(v)
    except (TypeError, ValueError):
        return str(v)


class CorrelationTable(QtWidgets.QTableWidget):
    def show_matrix(self, names, corr):
        self.clear()
        if corr is None:
            self.setRowCount(0)
            self.setColumnCount(0)
            return
        n = len(names)
        self.setRowCount(n)
        self.setColumnCount(n)
        self.setHorizontalHeaderLabels(names)
        self.setVerticalHeaderLabels(names)
        for i in range(n):
            for j in range(n):
                v = corr[i, j]
                it = QtWidgets.QTableWidgetItem(f"{v:+.2f}")
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                a = min(abs(v), 1.0)
                if v >= 0:
                    col = QtGui.QColor(int(255 - 120 * a), int(255 - 90 * a), 255)
                else:
                    col = QtGui.QColor(255, int(255 - 110 * a), int(255 - 130 * a))
                it.setBackground(col)
                if i != j and abs(v) > 0.9:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                    it.setToolTip("|ρ| > 0.9: parameters poorly determined individually")
                self.setItem(i, j, it)
        self.resizeColumnsToContents()


class ResultsPanel(QtWidgets.QWidget):
    profileRequested = QtCore.Signal()
    bootstrapRequested = QtCore.Signal()
    variantRequested = QtCore.Signal()
    compareRequested = QtCore.Signal()
    exportRequested = QtCore.Signal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.header = QtWidgets.QLabel("No fit yet.")
        self.header.setWordWrap(True)
        lay.addWidget(self.header)
        self.tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.tabs, 1)

        # statistics + warnings
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        self.stats = QtWidgets.QTableWidget(0, 2)
        self.stats.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.stats.horizontalHeader().setStretchLastSection(True)
        self.stats.verticalHeader().setVisible(False)
        self.stats.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        v.addWidget(self.stats, 3)
        self.warn = QtWidgets.QListWidget()
        self.warn.setWordWrap(True)
        v.addWidget(self.warn, 2)
        self.tabs.addTab(w, "Statistics")

        self.derived = QtWidgets.QTableWidget(0, 4)
        self.derived.setHorizontalHeaderLabels(["Component", "Quantity", "Value ± SE", "rel."])
        self.derived.horizontalHeader().setStretchLastSection(True)
        self.derived.verticalHeader().setVisible(False)
        self.derived.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.derived, "Derived")

        self.corr = CorrelationTable()
        self.tabs.addTab(self.corr, "Correlation")

        diag = pg.GraphicsLayoutWidget()
        self.p_hist = diag.addPlot(row=0, col=0, title="Histogram of normalized residuals")
        self.p_qq = diag.addPlot(row=0, col=1, title="Normal Q–Q plot")
        self.p_acf = diag.addPlot(row=1, col=0, colspan=2, title="Residuals vs. index")
        self.tabs.addTab(diag, "Residuals")

        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        f.setPointSize(9)
        self.report.setFont(f)
        self.tabs.addTab(self.report, "Report")

        row = QtWidgets.QHBoxLayout()
        for text, sig, tip in (("Profile CI", self.profileRequested, "Asymmetric confidence intervals from the "
                                                                      "profile likelihood"),
                               ("Bootstrap…", self.bootstrapRequested, "Residual or wild bootstrap"),
                               ("Remember variant", self.variantRequested, "Store this fit for model comparison"),
                               ("Compare…", self.compareRequested, "ΔAICc / ΔBIC / Akaike weights"),
                               ("Export…", self.exportRequested, "Tables, report, JSON")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(sig.emit)
            row.addWidget(b)
        lay.addLayout(row)
        state.currentChanged.connect(self.refresh)
        state.fitChanged.connect(lambda _id: self.refresh())
        state.modelChanged.connect(lambda _id: self.refresh())
        state.pipelineChanged.connect(lambda _id: self.refresh())
        state.fitOptionsChanged.connect(lambda _id: self.refresh())

    # ------------------------------------------------------------------ refresh
    def refresh(self):
        ds = self.state.current()
        r = ds.fit_result if ds is not None else None
        self.stats.setRowCount(0)
        self.warn.clear()
        self.derived.setRowCount(0)
        self.corr.show_matrix([], None)
        for p in (self.p_hist, self.p_qq, self.p_acf):
            p.clear()
        if r is None:
            self.header.setText("No fit yet. Build a model and click “Run fit” (Ctrl+R).")
            self.report.setPlainText("")
            return
        current = self.state.fit_current(ds)
        ok = "✓ converged" if r.success else "✖ not converged"
        stale = "" if current else "  <span style='color:#b36b00'>– out of date (data/model/options changed)</span>"
        self.header.setText(f"<b>{ok}</b> · {r.method} · {r.nfev} function evaluations{stale}")
        st = r.stats
        rows = [("N / p / ν", f"{st.n_points} / {st.n_varys} / {st.dof}"),
                ("σ source", st.sigma_label),
                ("Covariance", {"absolute": "absolute", "scaled": "scaled (by √χ²_ν or s)",
                                "unavailable": "not available"}[st.covariance_mode])]
        if st.chi2 is not None:
            lo, hi = st.redchi_band
            rows += [("χ²", _g(st.chi2, ".6g")),
                     ("χ²_ν", f"{_g(st.redchi, '.4g')}   (expected 1σ band {lo:.3g} … {hi:.3g})"),
                     ("P(χ² ≥ obs.) ≈", _g(st.chi2_pvalue, ".3g"))]
        else:
            rows += [("χ²", "undefined (σ unknown)")]
        rows += [("s", _g(st.s_res)), ("RMSE", _g(st.rmse)), ("RSS", _g(st.rss, ".6g")),
                 ("AICc", _g(st.aicc, ".6g")), ("AIC", _g(st.aic, ".6g")), ("BIC", _g(st.bic, ".6g")),
                 ("IC form", st.ic_form),
                 ("R²", f"{_g(st.r2, '.6f')}  (descriptive)"),
                 ("Runs test", f"{st.runs.get('runs')} runs, expected {_g(st.runs.get('expected'), '.1f')}, "
                               f"p(too few) = {_g(st.runs.get('p_too_few'), '.3g')}"),
                 ("ρ₁ (lag-1 autocorrelation)", _g(st.lag1_autocorr, ".3f")),
                 ("Durbin–Watson", _g(st.durbin_watson, ".3f")),
                 ("Condition number", _g(st.jacobian_condition, ".3g"))]
        self.stats.setRowCount(len(rows))
        for i, (k, val) in enumerate(rows):
            a = QtWidgets.QTableWidgetItem(k)
            tip = next((t for key, t in TIPS.items() if k.startswith(key)), None)
            if tip:
                a.setToolTip(tip)
            self.stats.setItem(i, 0, a)
            b = QtWidgets.QTableWidgetItem(str(val))
            if tip:
                b.setToolTip(tip)
            self.stats.setItem(i, 1, b)
        self.stats.resizeColumnToContents(0)
        if not r.warnings:
            self.warn.addItem("✓ no issues found")
        for w in sorted(r.warnings, key=lambda w: {"error": 0, "warning": 1, "info": 2}[w.severity]):
            it = QtWidgets.QListWidgetItem(f"{SEVERITY_ICON[w.severity]} {w.message}")
            it.setForeground(QtGui.QColor(SEVERITY_COLOR[w.severity]))
            it.setToolTip(w.code)
            self.warn.addItem(it)
        # derived
        self.derived.setRowCount(len(r.derived))
        for i, d in enumerate(r.derived):
            rel = "" if d.stderr is None or d.value == 0 else f"{100 * abs(d.stderr / d.value):.2g} %"
            for j, txt in enumerate([d.component, d.name, fmt_value(d.value, d.stderr), rel]):
                self.derived.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        self.derived.resizeColumnsToContents()
        self.corr.show_matrix(r.var_names, r.correlation)
        # diagnostics
        z = np.asarray(r.normalized_residuals, float)
        if len(z) > 2:
            hist, edges = np.histogram(z, bins=max(10, min(60, len(z) // 15)), density=True)
            self.p_hist.plot(edges, hist, stepMode="center", fillLevel=0, brush=(63, 144, 218, 90),
                             pen=pg.mkPen("#3f90da"))
            t = np.linspace(min(-4, z.min()), max(4, z.max()), 200)
            self.p_hist.plot(t, np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi), pen=pg.mkPen("#bd1f01", width=1.5))
            qx, qy = S.qq_points(z)
            self.p_qq.plot(qx, qy, pen=None, symbol="o", symbolSize=3, symbolBrush="#3c3c3c", symbolPen=None)
            lim = [qx.min(), qx.max()]
            self.p_qq.plot(lim, lim, pen=pg.mkPen("#bd1f01"))
            self.p_qq.setLabel("bottom", "theoretical quantiles")
            self.p_qq.setLabel("left", "observed quantiles")
            self.p_acf.plot(np.arange(len(z)), z, pen=pg.mkPen("#3c3c3c", width=1))
            self.p_acf.addLine(y=0, pen=pg.mkPen("#808080"))
        self.report.setPlainText(r.report())
