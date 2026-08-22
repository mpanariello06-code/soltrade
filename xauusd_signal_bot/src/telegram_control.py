"""Telegram inline-button control panel.

Buttons are the primary interface; the only text commands are ``/start``,
``/panel`` and ``/menu``, which exist purely to summon the panel in the first
place.  Every press mutates :class:`~src.runtime_state.RuntimeState` in the
running process - nothing here requires a restart.

THREADING
---------
:meth:`TelegramController.start` runs a daemon thread that long-polls
``getUpdates``.  It shares the runtime state (lock-guarded) and the MT5
connector (lock-guarded) with the main signal loop, so a button press can be
serviced while the loop is mid-poll.

SAFETY
------
* Updates from any chat other than the configured ``TELEGRAM_CHAT_ID`` are
  ignored outright.
* Only settings listed in ``Config.telegram_editable_settings`` are reachable;
  credentials, the bot token, the symbol and file paths have no handler and are
  never rendered into a message.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logger import get_logger
from .runtime_state import RuntimeState
from .markets import MARKET_ORDER, get_market, normalise_market
from .timeframes import (
    MODE_SCALPING,
    SIGNAL_TIMEFRAME,
    STATUS_PAUSED,
    STATUS_RUNNING,
    STATUS_STOPPED,
)

LOGGER = get_logger("telegram.control")

DIVIDER = "━━━━━━━━━━━━━━━━━━"

STATUS_ICONS = {STATUS_RUNNING: "🟢", STATUS_PAUSED: "⏸", STATUS_STOPPED: "🔴"}

#: Component key -> label, in the order shown in the ANALYSIS breakdown.
#: The maximum for each row is read from the configured weights, so retuning
#: them for M1 cannot leave the panel quoting stale denominators.
ANALYSIS_ROWS: Tuple[Tuple[str, str], ...] = (
    ("momentum", "Momentum"),
    ("price_action", "Price Action"),
    ("liquidity", "Liquidity"),
    ("structure", "Structure"),
    ("support_resistance", "S/R"),
    ("trend", "Trend"),
    ("htf", "Context"),
    ("volatility", "Volatility"),
    ("volume", "Volume"),
)

COMMAND_WORDS = ("/start", "/panel", "/menu", "/status")


def _button(text: str, data: str) -> Dict[str, str]:
    """One inline keyboard button."""
    return {"text": text, "callback_data": data}


class TelegramController:
    """Renders the panel and routes button presses onto the runtime state.

    ``engine`` is optional and only needs to provide what the panel asks for:

    ``analyze_now()``     -> an :class:`~src.signal_engine.Evaluation` or ``None``
    ``signals_today()``   -> int
    ``connection_state()``-> str
    """

    def __init__(
        self,
        config,
        runtime: RuntimeState,
        notifier,
        engine: Any = None,
        report_loader: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.notifier = notifier
        self.engine = engine
        self._report_loader = report_loader
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._offset = 0
        self._panel_message_id: Optional[int] = None
        #: Market the PERFORMANCE view is scoped to; "COMBINED" merges them.
        #: Purely UI state - it never affects which market is being evaluated.
        self._perf_symbol: Optional[str] = None

    # ------------------------------------------------------------------ #
    # keyboards
    # ------------------------------------------------------------------ #
    def main_keyboard(self) -> List[List[Dict[str, str]]]:
        """The main control panel keyboard.

        The market row marks the active market with a filled dot.  START/PAUSE
        is one toggle rather than two buttons, so the panel shows the action
        that is actually available.
        """
        active = self.runtime.active_market
        market_row = []
        for symbol in MARKET_ORDER:
            market = get_market(symbol)
            mark = "●" if symbol == active else "○"
            market_row.append(_button(f"{mark} {market.icon} {symbol}", f"mkt:{symbol}"))

        running = self.runtime.is_running
        toggle = (
            _button("⏸ PAUSE", "run:pause") if running else _button("▶️ START", "run:start")
        )
        return [
            market_row,
            [_button("📊 CURRENT ANALYSIS", "view:analysis")],
            [_button("📈 PERFORMANCE", "view:performance")],
            [toggle],
            [_button("⚙️ SETTINGS", "menu:settings")],
            [_button("🔄 REFRESH", "panel:refresh")],
        ]

    def threshold_keyboard(self) -> List[List[Dict[str, str]]]:
        steps = self.config.threshold_steps
        return [
            [_button(f"{step:+d}", f"thr:{step}") for step in steps],
            [_button("♻️ RESET", "thr:reset")],
            [_button("⬅️ BACK", "menu:settings")],
        ]

    def settings_keyboard(self) -> List[List[Dict[str, str]]]:
        """Settings always act on the ACTIVE market - never on the other one."""
        state = self.runtime.describe()
        near = "ON ✅" if state["near_signal_alerts"] else "OFF"
        return [
            [_button(f"🎯 Threshold: {state['threshold']:.0f}", "menu:threshold")],
            [_button(f"⏳ Holding time: {state['max_holding_candles']} min", "set:hold")],
            [_button(f"❄️ Cooldown: {state['cooldown_candles']} min", "set:cooldown")],
            [_button(f"⚖️ Min R:R: {state['min_tp2_rr']:.1f}", "set:rr")],
            [_button(f"🕐 Sessions: {', '.join(state['allowed_sessions'])[:22]}", "set:session")],
            [_button(f"👀 Near-signal alerts: {near}", "set:near")],
            [_button("⏹ STOP ENGINE", "run:stop")],
            [_button("⬅️ BACK", "nav:main")],
        ]

    def performance_keyboard(self) -> List[List[Dict[str, str]]]:
        """Market selector plus the per-market drill-downs.

        Markets are deliberately NOT merged by default: an R on gold and an R on
        Bitcoin share a unit but come from different cost and volatility
        regimes.  COMBINED is offered and clearly labelled as such.
        """
        scope = self._perf_scope()
        row = []
        for symbol in MARKET_ORDER:
            market = get_market(symbol)
            mark = "●" if symbol == scope else "○"
            row.append(_button(f"{mark} {market.icon} {symbol}", f"perf:{symbol}"))
        combined_mark = "●" if scope == "COMBINED" else "○"
        return [
            row,
            [_button(f"{combined_mark} 📊 COMBINED", "perf:COMBINED")],
            [_button("BY SCORE", "perfv:score"), _button("BY REGIME", "perfv:regime")],
            [_button("BY HOUR", "perfv:hour"), _button("BY SESSION", "perfv:session")],
            [_button("BY OUTCOME", "perfv:outcome")],
            [_button("⬅️ BACK", "nav:main")],
        ]

    @staticmethod
    def back_keyboard() -> List[List[Dict[str, str]]]:
        return [[_button("⬅️ BACK", "nav:main")]]

    # ------------------------------------------------------------------ #
    # panel rendering
    # ------------------------------------------------------------------ #
    def render_panel(self) -> str:
        """The main control panel text - active market stated prominently."""
        state = self.runtime.describe()
        status = state["status"]
        custom = " (custom)" if state["threshold_is_custom"] else ""
        symbol = state["active_market"]
        net_r = self._paper_net_r(symbol)

        lines = [
            DIVIDER,
            "⚡ M1 SCALPER",
            DIVIDER,
            "",
            f"Market: {state['label']}",
            f"Timeframe: {SIGNAL_TIMEFRAME}",
            f"Mode: {MODE_SCALPING}",
            f"Status: {STATUS_ICONS.get(status, '❔')} {status}",
            "",
            f"Threshold: {state['threshold']:.0f}{custom}",
            f"Max hold: {state['max_holding_candles']} min",
            f"Cost model: {state['cost_model']}",
            "",
            f"Signals today: {self._signals_today(symbol)}",
            f"Open signals: {self._open_signals(symbol)}",
            f"Paper Net R: {net_r:+.2f}" if net_r is not None else "Paper Net R: -",
            "",
            f"MT5: {self._connection_state()}",
        ]
        other = [s for s in MARKET_ORDER if s != symbol]
        if other:
            carried = ", ".join(
                f"{get_market(o).icon} {o} {self._open_signals(o)} open" for o in other
            )
            lines += ["", f"Also tracking: {carried}"]
        lines += ["", DIVIDER, "🔬 PAPER TEST ONLY - no orders are placed."]
        # Driven by the market's own note, not a symbol comparison: a hardcoded
        # name silently stops matching the moment a symbol is renamed.
        note = get_market(symbol).note
        if "INITIAL RESEARCH PARAMETERS" in note.upper():
            lines.append(f"{symbol} uses INITIAL RESEARCH PARAMETERS.")
        return "\n".join(lines)

    def render_threshold_panel(self) -> str:
        state = self.runtime.describe()
        return "\n".join(
            [
                DIVIDER,
                f"🎯 {state['label']} THRESHOLD",
                DIVIDER,
                "",
                f"Current threshold: {state['threshold']:.0f}",
                f"Default for {state['symbol']}: {state['default_threshold']:.0f}",
                "",
                f"Allowed range: {self.config.min_threshold:.0f} - {self.config.max_threshold:.0f}",
                "",
                "The regime still adjusts this per candle;",
                "this is the base value.",
                "",
                "Thresholds are per market - changing this",
                "does not affect the other market.",
                DIVIDER,
            ]
        )

    def render_settings(self) -> str:
        """Settings for the ACTIVE market only."""
        state = self.runtime.describe()
        market = get_market(state["symbol"])
        view = self.config.for_market(market)
        cost = view.pips(view.round_trip_cost(None))
        return "\n".join(
            [
                DIVIDER,
                f"⚙️ {state['symbol']} SETTINGS",
                DIVIDER,
                "",
                f"Market: {state['label']}",
                f"Timeframe: {SIGNAL_TIMEFRAME}  (fixed)",
                f"Mode: {MODE_SCALPING}  (fixed)",
                f"24/7 market: {'yes' if state['is_24h'] else 'no'}",
                "",
                f"Threshold: {state['threshold']:.0f}",
                f"Max holding period: {state['max_holding_candles']} minutes",
                f"Cooldown: {state['cooldown_candles']} candles",
                f"Minimum R:R (TP2): {state['min_tp2_rr']:.1f}",
                f"Session filter: {', '.join(state['allowed_sessions'])}"
                + ("  (not applied - 24/7)" if state["is_24h"] else ""),
                f"Near-signal alerts: {'ON' if state['near_signal_alerts'] else 'OFF'}",
                "",
                f"Cost model: {state['cost_model']}",
                "Costs assumed per round trip:",
                f"  {cost:.1f}{state['pip_name']}"
                f"  (spread {view.assumed_spread_points:.0f} + slippage "
                f"{view.slippage_points_entry:.0f}+{view.slippage_points_exit:.0f} pts)",
                "",
                "These settings apply to this market only.",
                "Credentials and file paths are not editable here.",
                DIVIDER,
            ]
        )

    def render_analysis(self) -> str:
        """Latest evaluation, computed on demand from closed candles only."""
        symbol = self.runtime.active_market
        evaluation = self._analyze_now(symbol)
        if evaluation is None:
            return "\n".join(
                [DIVIDER, f"📊 {symbol} M1 ANALYSIS", DIVIDER, "",
                 "No market data available right now.",
                 "(MT5 disconnected, or not enough closed candles yet.)", DIVIDER]
            )

        card = evaluation.card
        if card is None:
            return "\n".join(
                [DIVIDER, "📊 ANALYSIS", DIVIDER, "",
                 f"No evaluation: {evaluation.rejection_reason or 'unavailable'}", DIVIDER]
            )

        direction = "BUY" if card.bullish_score >= card.bearish_score else "SELL"
        lines = [
            DIVIDER,
            f"📊 {get_market(evaluation.symbol).icon} {evaluation.symbol} M1 ANALYSIS",
            DIVIDER,
            "",
            f"Bullish Score: {card.bullish_score:.0f}",
            f"Bearish Score: {card.bearish_score:.0f}",
            "",
            "Regime:",
            evaluation.regime.replace("_", " ") or "-",
            "",
        ]
        for key, label in ANALYSIS_ROWS:
            maximum = getattr(self.config.weights, key, 0.0)
            value = card.component_score(key, direction)
            component = card.components.get(key)
            if component is not None and not component.applicable:
                lines.append(f"{label + ':':<14}  n/a")
            else:
                lines.append(f"{label + ':':<14}{value:5.1f}/{maximum:.0f}")

        features = evaluation.features or {}
        lines += [
            "",
            f"Threshold: {evaluation.threshold:.0f}",
            f"ATR: {features.get('atr_pips', 0):.1f}p   "
            f"Spread: {features.get('spread_pips', 0):.1f}p   "
            f"Cost: {features.get('cost_pips', 0):.1f}p",
            f"Candle: {evaluation.timestamp:%Y-%m-%d %H:%M} UTC",
            "",
            "Decision:",
        ]
        if evaluation.has_signal and evaluation.signal is not None:
            signal = evaluation.signal
            lines += [
                f"⚡ {signal.direction} SCALP",
                f"TP {signal.tp_pips[0]:.1f}/{signal.tp_pips[1]:.1f}/{signal.tp_pips[2]:.1f}p"
                f"  SL {signal.sl_pips:.1f}p",
                f"Net R:R  {signal.net_rr1:.2f}/{signal.net_rr2:.2f}/{signal.net_rr3:.2f}",
            ]
        elif evaluation.near_signal:
            best = max(card.bullish_score, card.bearish_score)
            lines += [
                "👀 NEAR SIGNAL",
                "",
                "Reason:",
                f"{direction} score {best:.0f} < threshold {evaluation.threshold:.0f}",
            ]
        else:
            lines += ["❌ NO SIGNAL", "", "Reason:", evaluation.rejection_reason or "-"]

        lines += ["", DIVIDER, "Closed M1 candles only. No forward-looking data."]
        return "\n".join(lines)

    # -- performance views -------------------------------------------------- #
    def _perf_scope(self) -> str:
        """Market the performance view is scoped to (defaults to the active one)."""
        return self._perf_symbol or self.runtime.active_market

    def _report(self, symbol: Optional[str] = None):
        if self._report_loader is None:
            return None
        try:
            return self._report_loader(symbol)
        except TypeError:
            # a loader that predates per-market reports
            try:
                return self._report_loader()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Could not build the performance report: %s", exc)
                return None
        except Exception as exc:  # noqa: BLE001 - a report must never break the panel
            LOGGER.exception("Could not build the performance report: %s", exc)
            return None

    @staticmethod
    def _stats_table(title: str, rows) -> List[str]:
        if not rows:
            return [title, "  (no closed signals yet)"]
        lines = [title, f"  {'group':<22}{'n':>4}{'win%':>7}{'avgR':>8}{'PF':>7}"]
        for stats in rows:
            profit_factor = (
                "inf" if stats.net_profit_factor == float("inf")
                else f"{stats.net_profit_factor:.2f}"
            )
            lines.append(
                f"  {stats.label[:21]:<22}{stats.closed:>4}{stats.net_win_rate:>7.1f}"
                f"{stats.average_net_r:>+8.2f}{profit_factor:>7}"
            )
        return lines

    def render_performance(self, view: str = "summary") -> str:
        """Paper-performance for one market, read straight from its CSV files."""
        scope = self._perf_scope()
        report = self._report(scope)
        title = "📊 COMBINED" if scope == "COMBINED" else get_market(scope).label()

        if report is None or (report.total_signals == 0 and report.total_evaluations == 0):
            return "\n".join(
                [DIVIDER, f"📈 {title} PERFORMANCE", DIVIDER, "",
                 "No signals recorded yet.", DIVIDER]
            )

        overall = report.overall
        header = [
            DIVIDER,
            f"📈 {title} PERFORMANCE",
            DIVIDER,
            "",
            f"{SIGNAL_TIMEFRAME} {MODE_SCALPING}",
            "",
        ]

        if view == "score":
            body = self._stats_table("BY SCORE BAND", report.by_score_band)
            body += ["", "Higher score = better outcome?  Check, do not assume."]
        elif view == "regime":
            body = self._stats_table("BY REGIME", report.by_regime)
        elif view == "hour":
            body = self._stats_table("BY UTC HOUR", report.by_hour)
        elif view == "session":
            body = self._stats_table("BY SESSION", report.by_session)
            if scope != "COMBINED" and get_market(scope).is_24h:
                body += ["", "This market trades 24/7 - the session label is",
                         "recorded for analysis but never filters signals."]
        elif view == "outcome":
            body = self._stats_table("BY OUTCOME", report.by_result)
        else:
            profit_factor = (
                "inf" if overall.net_profit_factor == float("inf")
                else f"{overall.net_profit_factor:.2f}"
            )
            body = [
                f"Evaluations: {report.total_evaluations}",
                f"Signals: {report.total_signals}",
                f"Closed & scored: {overall.closed}",
                f"Wins / Losses: {overall.wins} / {overall.losses}",
                f"Timed out: {overall.timeout_rate:.1f}%",
                "",
                f"Win rate (net): {overall.net_win_rate:.1f}%",
                f"Avg RAW R: {overall.average_r:+.3f}",
                f"Avg NET R: {overall.average_net_r:+.3f}",
                f"Total NET R: {overall.total_net_r:+.2f}",
                f"Profit factor (net): {profit_factor}",
                f"Max drawdown: {overall.max_drawdown_r:.2f}R",
                f"Avg holding time: {overall.average_duration_min:.1f} min",
                f"Avg cost: {overall.average_cost_r:.2f}R",
                "",
                f"TP1 reached: {overall.tp1_rate:.1f}%   SL: {overall.sl_rate:.1f}%",
                "",
                "Best Regime:",
                report.best_regime(),
                "",
                "Worst Regime:",
                report.worst_regime(),
            ]
            if scope == "COMBINED":
                body += ["", "⚠️ Markets merged - different cost and",
                         "volatility regimes.  Per-market is the honest view."]
        return "\n".join(header + body + ["", DIVIDER, "Paper results only. Not advice."])

    # ------------------------------------------------------------------ #
    # callback routing
    # ------------------------------------------------------------------ #
    def handle_callback(self, data: str) -> Tuple[str, List[List[Dict[str, str]]], str]:
        """Route one button press.

        Returns ``(text, keyboard, toast)`` - the panel body, its keyboard, and
        a short confirmation shown as a Telegram toast.
        """
        data = str(data or "").strip()
        action, _, argument = data.partition(":")

        try:
            if action == "run":
                return self._handle_run(argument)
            if action == "mkt":
                return self._handle_market(argument)
            if action == "thr":
                return self._handle_threshold(argument)
            if action == "menu":
                return self._handle_menu(argument)
            if action == "set":
                return self._handle_setting(argument)
            if action == "view":
                if argument == "analysis":
                    return self.render_analysis(), self.back_keyboard(), "Analysing…"
                if argument == "performance":
                    self._perf_symbol = self.runtime.active_market
                    return (
                        self.render_performance(),
                        self.performance_keyboard(),
                        "Performance",
                    )
            if action == "perf":
                # scope the performance view to a market (or COMBINED)
                self._perf_symbol = (
                    "COMBINED" if argument.upper() == "COMBINED"
                    else normalise_market(argument, self.runtime.active_market)
                )
                return (
                    self.render_performance(),
                    self.performance_keyboard(),
                    self._perf_symbol,
                )
            if action == "perfv":
                return (
                    self.render_performance(argument),
                    self.performance_keyboard(),
                    argument,
                )
            if action in ("nav", "panel"):
                return self.render_panel(), self.main_keyboard(), "Refreshed"
        except Exception as exc:  # noqa: BLE001 - a bad press must not kill the thread
            LOGGER.exception("Callback '%s' failed: %s", data, exc)
            return self.render_panel(), self.main_keyboard(), "Something went wrong"

        return self.render_panel(), self.main_keyboard(), ""

    def _handle_market(self, argument: str):
        """Switch the active market.

        Selection only: the other market's state, CSV files, cooldowns and open
        paper trades are untouched, and its open trades keep being tracked.
        """
        previous = self.runtime.active_market
        symbol = self.runtime.set_active_market(argument)
        self._perf_symbol = None
        if symbol != previous:
            notify = getattr(self.engine, "on_market_changed", None)
            if callable(notify):
                notify(previous, symbol)
        market = get_market(symbol)
        return self.render_panel(), self.main_keyboard(), f"Market: {market.label()}"

    def _handle_run(self, argument: str):
        mapping = {
            "start": self.runtime.start,
            "pause": self.runtime.pause,
            "stop": self.runtime.stop,
        }
        handler = mapping.get(argument)
        if handler is None:
            return self.render_panel(), self.main_keyboard(), ""
        status = handler()
        toast = {
            STATUS_RUNNING: "Engine running",
            STATUS_PAUSED: "Paused - no new signals",
            STATUS_STOPPED: "Stopping…",
        }.get(status, status)
        return self.render_panel(), self.main_keyboard(), toast

    def _handle_threshold(self, argument: str):
        """Threshold controls always act on the ACTIVE market."""
        market_runtime = self.runtime.active
        if argument == "reset":
            value = market_runtime.reset_threshold()
            toast = f"{market_runtime.symbol} reset to {value:.0f}"
        else:
            try:
                value = market_runtime.adjust_threshold(float(argument))
            except ValueError:
                value = market_runtime.active_threshold()
            toast = f"{market_runtime.symbol} threshold: {value:.0f}"
        return self.render_threshold_panel(), self.threshold_keyboard(), toast

    def _handle_menu(self, argument: str):
        if argument == "threshold":
            return self.render_threshold_panel(), self.threshold_keyboard(), "Threshold"
        if argument == "settings":
            return self.render_settings(), self.settings_keyboard(), "Settings"
        return self.render_panel(), self.main_keyboard(), ""

    def _handle_setting(self, argument: str):
        """Cycle one editable setting of the ACTIVE market to its next value."""
        state = self.runtime.describe()
        market_runtime = self.runtime.active
        symbol = market_runtime.symbol

        def cycle(choices, current):
            choices = list(choices)
            if current in choices:
                return choices[(choices.index(current) + 1) % len(choices)]
            return choices[0]

        if argument == "cooldown":
            value = cycle(self.config.cooldown_choices, int(state["cooldown_candles"]))
            market_runtime.set_cooldown(value)
            toast = f"{symbol} cooldown: {value} candles"
        elif argument == "rr":
            value = cycle(self.config.min_rr_choices, float(state["min_tp2_rr"]))
            market_runtime.set_min_rr(value)
            toast = f"{symbol} min R:R: {value:.1f}"
        elif argument == "hold":
            value = cycle(self.config.holding_choices, int(state["max_holding_candles"]))
            market_runtime.set_max_holding(value)
            toast = f"{symbol} max hold: {value} min"
        elif argument == "session":
            current = ",".join(state["allowed_sessions"])
            value = cycle(self.config.session_choices, current)
            market_runtime.set_sessions([value])
            toast = f"{symbol} sessions: {value}"
        elif argument == "near":
            value = self.runtime.set_near_signal_alerts(not state["near_signal_alerts"])
            toast = f"Near-signal alerts: {'ON' if value else 'OFF'}"
        else:
            toast = ""
        return self.render_settings(), self.settings_keyboard(), toast

    # ------------------------------------------------------------------ #
    # engine hooks
    # ------------------------------------------------------------------ #
    def _analyze_now(self, symbol: Optional[str] = None):
        analyse = getattr(self.engine, "analyze_now", None)
        if not callable(analyse):
            return None
        try:
            return analyse(symbol)
        except TypeError:
            try:
                return analyse()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("On-demand analysis failed: %s", exc)
                return None
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("On-demand analysis failed: %s", exc)
            return None

    def _call_engine(self, name: str, symbol: Optional[str], default=0):
        """Call an optional engine hook that may or may not take a symbol."""
        hook = getattr(self.engine, name, None)
        if not callable(hook):
            return default
        try:
            return hook(symbol)
        except TypeError:
            try:
                return hook()
            except Exception:  # noqa: BLE001
                return default
        except Exception:  # noqa: BLE001
            return default

    def _open_signals(self, symbol: Optional[str] = None) -> int:
        try:
            return int(self._call_engine("open_signals", symbol, 0))
        except (TypeError, ValueError):
            return 0

    def _paper_net_r(self, symbol: Optional[str] = None) -> Optional[float]:
        """Cumulative NET R from this market's closed paper signals."""
        report = self._report(symbol or self.runtime.active_market)
        if report is None or report.overall.closed == 0:
            return None
        return report.overall.total_net_r

    def _signals_today(self, symbol: Optional[str] = None) -> int:
        try:
            return int(self._call_engine("signals_today", symbol, 0))
        except (TypeError, ValueError):
            return 0

    def _connection_state(self) -> str:
        state = getattr(self.engine, "connection_state", None)
        try:
            return str(state()) if callable(state) else "unknown"
        except Exception:  # noqa: BLE001
            return "unknown"

    # ------------------------------------------------------------------ #
    # polling loop
    # ------------------------------------------------------------------ #
    def send_panel(self) -> Optional[int]:
        """Post a fresh control panel and remember it for in-place editing."""
        message_id = self.notifier.send_message(self.render_panel(), self.main_keyboard())
        if message_id:
            self._panel_message_id = message_id
        return message_id

    def _show(self, message_id: Optional[int], text: str, keyboard) -> None:
        """Edit the pressed message in place, falling back to a new message."""
        if message_id and self.notifier.edit_message(message_id, text, keyboard):
            self._panel_message_id = message_id
            return
        new_id = self.notifier.send_message(text, keyboard)
        if new_id:
            self._panel_message_id = new_id

    def _authorised(self, chat_id: Any) -> bool:
        """Only the configured chat may drive the engine."""
        return str(chat_id) == str(self.config.telegram_chat_id)

    def process_update(self, update: Dict[str, Any]) -> None:
        """Handle one Telegram update object."""
        callback = update.get("callback_query")
        if callback:
            message = callback.get("message") or {}
            chat_id = (message.get("chat") or {}).get("id")
            if not self._authorised(chat_id):
                LOGGER.warning("Ignoring callback from unauthorised chat %s", chat_id)
                return
            text, keyboard, toast = self.handle_callback(callback.get("data", ""))
            self.notifier.answer_callback(callback.get("id", ""), toast)
            self._show(message.get("message_id"), text, keyboard)
            return

        message = update.get("message")
        if message:
            chat_id = (message.get("chat") or {}).get("id")
            if not self._authorised(chat_id):
                return
            text = str(message.get("text", "")).strip().lower().split("@")[0]
            if text in COMMAND_WORDS:
                self.send_panel()

    def _loop(self) -> None:
        LOGGER.info("Telegram control panel listening")
        while not self._stop_event.is_set():
            try:
                updates = self.notifier.get_updates(
                    self._offset, timeout=self.config.telegram_poll_timeout
                )
                for update in updates:
                    self._offset = max(self._offset, int(update.get("update_id", 0)) + 1)
                    self.process_update(update)
                if not updates:
                    # nothing pending; a short pause keeps a failing long-poll
                    # from becoming a busy loop
                    self._stop_event.wait(1.0)
            except Exception as exc:  # noqa: BLE001 - the poller must never die
                LOGGER.exception("Telegram poll failed: %s", exc)
                self._stop_event.wait(5.0)
        LOGGER.info("Telegram control panel stopped")

    def start(self) -> bool:
        """Start the background poller.  Returns ``False`` when disabled."""
        if not self.config.telegram_control_enabled or not self.notifier.enabled:
            LOGGER.info("Telegram control panel disabled")
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._drain_backlog()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="telegram-control", daemon=True)
        self._thread.start()
        return True

    def _drain_backlog(self) -> None:
        """Skip updates queued while the engine was down.

        Replaying old button presses on startup would apply stale commands - the
        last thing a user expects after a restart.
        """
        updates = self.notifier.get_updates(0, timeout=0)
        if updates:
            self._offset = max(int(u.get("update_id", 0)) for u in updates) + 1
            LOGGER.info("Skipped %s queued Telegram update(s) from before startup", len(updates))

    def stop(self) -> None:
        """Stop the poller thread."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(self.config.telegram_poll_timeout + 5, 5))
        self._thread = None
