"""
事件合约方向信号系统 v5.3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三层评分 + 双品种 + 信号历史胜率统计

第一层（1m K线）：RSI(14) / 放量 / VWAP / BB / EMA共振 / 整数关口
第二层（实时API）：OBI / 持续 / 加速 / 深度斜率 / Taker / CVD / 大单 / 费率 / OI / 溢价
第三层（5m趋势）：EMA(7/21) / RSI(14) / BB — 方向过滤（顺+3分，逆-8分）

v5.3改动：
  MIN_CONF: 40→75
  BB mult: 2.5→2.0，权重4→7
  RSI_VOL: vr>1.5→vr>1.0，权重12→15
  STOCH_RSI: 权重4→1（95%极值无效）
  PV_DIV: 权重3→0（触发0.5%废弃）
  RSI区间: [25,75]→[30,70]
  新增ATR波动率门槛：过滤低波动噪音单
  新增Layer3(5m趋势)方向过滤

信号记录 + 延迟验证胜率

修复日志 v5.1:
  [Bug1] _rsi(): Wilder平滑方向反了——种子改为首period均值，迭代改为从前往后
  [Bug2] evaluate(): 置信度4套常量实际只用了CONF_AGREE_*，按agree_label分支选取
  [Bug3] Layer2.score(): OI变化 oi_s 永远赋0，实现结合价格方向的OI信号
  [Bug4] Layer2.collect(): 清算endpoint /fapi/v1/forceOrders需认证，改为公开接口
         /fapi/v1/allForceOrders，字段名同步修正(origQty/price)
  [Bug5] spot_sym: self.symbol.replace("USDT","")+... 无效变换，改为取base币种
  [Fix6] 去重逻辑: 同品种反向信号不受5分钟冷却限制，只有同方向才冷却
  [Fix7] CVD历史: 检查>=5条但取[-10:]不一致，统一为取实际可用数量
  [Fix8] _init_history(): 改用session连接池而非裸requests.get
  [Fix9] score(): VWAP_DEV补充独立details条目
  [Fix10] SignalHistory: 加threading.Lock防止并发读写竞争
  [Fix11] 关键except不再静默吞错，改为打印警告
  [Fix12] _magnet(): 步长动态按价格数量级计算，不再硬编码1000/100
"""

import requests
import numpy as np
import json
import os
import time
import sys
import threading
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ───────────────────────────────────────────────
# 日志（Fix11）
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("signal")

# 连接池复用，减少TCP握手开销
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10)
session.mount("https://", adapter)

# ───────────────────────────────────────────────
# 配置
# ───────────────────────────────────────────────
SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}
FAPI = "https://fapi.binance.com"
SAPI = "https://api.binance.com"

COLLECT_INTERVAL = 5
REPORT_EVERY = 1
SETTLE_MINS = 10
MIN_CONF = 80  # 最低置信度，低于此值不显示/不记录信号
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal_history.json")

# 第一层权重（K线指标）
L1_WEIGHTS = {
    "RSI14":       25,   # 核心，回测58.9%
    "RSI_VOL":     15,   # RSI+放量（vr>1.0触发，权重提升）
    "RSI_VWAP":     8,   # RSI+VWAP偏离
    "STOCH_RSI":    1,   # StochRSI（95%极值，降权）
    "BB":           7,   # 布林带（mult=2.0，触发率提升）
    "EMA_RESON":    4,   # EMA共振
    "PV_DIV":       0,   # 价量背离（触发0.5%，废弃）
    "MAGNET":       2,   # 整数关口
    "VWAP_DEV":     3,   # VWAP偏离独立
}

# 第二层权重（实时API）
L2_WEIGHTS = {
    "OBI":         10,
    "OBI_PERSIST":  4,
    "OBI_SPEED":    3,
    "DEPTH_SLOPE":  3,
    "TAKER":        5,
    "CVD":          3,
    "LARGE_ORDER":  3,
    "LIQUIDATION":  3,
    "FUNDING":      2,
    "OI_CHANGE":    2,
    "PERP_PREM":    2,
}

L1_MAX = sum(L1_WEIGHTS.values())  # 65
L2_MAX = sum(L2_WEIGHTS.values())  # 40

# 置信度参数（0-100，修改这里调整置信度计算）
CONF_AGREE_BASE    = 60    # L1/L2方向一致时基础分
CONF_AGREE_MULT    = 0.8   # 一致时信号强度系数
CONF_ONLYL1_BASE   = 40    # 仅L1有信号时基础分
CONF_ONLYL1_MULT   = 0.5   # 仅L1时信号强度系数
CONF_ONLYL2_BASE   = 35    # 仅L2有信号时基础分
CONF_ONLYL2_MULT   = 0.45  # 仅L2时信号强度系数
CONF_CONFLICT_BASE = 20    # L1/L2冲突时基础分
CONF_CONFLICT_MULT = 0.3   # 冲突时信号强度系数
CONF_NONE_BASE     = 10    # 无信号时基础分
CONF_NONE_MULT     = 0.2   # 无信号时信号强度系数


# ───────────────────────────────────────────────
# API 工具
# ───────────────────────────────────────────────
def api_get(base, path, params, timeout=3):
    for attempt in range(3):
        r = session.get(f"{base}{path}", params=params, timeout=timeout)
        if r.status_code == 429:
            wait = 2 ** attempt
            logger.warning(f"429限流，等待{wait}秒后重试")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


# ───────────────────────────────────────────────
# 第一层：K线指标
# ───────────────────────────────────────────────
class Layer1:
    def __init__(self, symbol, lookback=100):
        self.symbol = symbol
        self.lookback = lookback
        self.closes = deque(maxlen=lookback)
        self.highs = deque(maxlen=lookback)
        self.lows = deque(maxlen=lookback)
        self.opens = deque(maxlen=lookback)
        self.volumes = deque(maxlen=lookback)
        self.taker_buy_vols = deque(maxlen=lookback)
        self._init_history()

    def _init_history(self):
        # [Fix8] 使用全局 session 连接池，避免每次建立新TCP连接
        resp = session.get(f"{FAPI}/fapi/v1/klines", params={
            "symbol": self.symbol, "interval": "1m", "limit": self.lookback
        }, timeout=10)
        resp.raise_for_status()
        for k in resp.json():
            self.closes.append(float(k[4]))
            self.highs.append(float(k[2]))
            self.lows.append(float(k[3]))
            self.opens.append(float(k[1]))
            self.volumes.append(float(k[5]))
            self.taker_buy_vols.append(float(k[9]))

    def update(self, close, high, low, open_p, volume, taker_buy_vol):
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.opens.append(open_p)
        self.volumes.append(volume)
        self.taker_buy_vols.append(taker_buy_vol)

    def _rsi(self, period=14):
        """
        [Bug1 Fix] Wilder平滑RSI
        原代码错误：种子取g[-period:]（最后period个），然后从后往前迭代
        正确做法：种子取g[:period]（最前period个），然后从前往后迭代
        与同文件_stoch_rsi()的实现保持一致
        """
        if len(self.closes) < period + 1:
            return 50
        c = np.array(self.closes)
        d = np.diff(c)
        gains = np.where(d > 0, d, 0)
        losses = np.where(d < 0, -d, 0)
        # 种子：首个period的简单均值
        ag = np.mean(gains[:period])
        al = np.mean(losses[:period])
        # 正向 Wilder 平滑
        for i in range(period, len(gains)):
            ag = (ag * (period - 1) + gains[i]) / period
            al = (al * (period - 1) + losses[i]) / period
        if al == 0:
            return 100
        return 100 - 100 / (1 + ag / al)

    def _stoch_rsi(self, rsi_p=14, stoch_p=14):
        if len(self.closes) < rsi_p + stoch_p + 1:
            return 50
        c = np.array(self.closes)
        d = np.diff(c)
        gains = np.where(d > 0, d, 0)
        losses = np.where(d < 0, -d, 0)
        rsi_vals = []
        ag = np.mean(gains[:rsi_p])
        al = np.mean(losses[:rsi_p])
        for i in range(rsi_p, len(gains)):
            ag = (ag * (rsi_p - 1) + gains[i]) / rsi_p
            al = (al * (rsi_p - 1) + losses[i]) / rsi_p
            rsi_vals.append(100 - 100 / (1 + ag / al) if al > 0 else 100)
        if len(rsi_vals) < stoch_p:
            return 50
        recent = rsi_vals[-stoch_p:]
        mn, mx = min(recent), max(recent)
        if mx == mn:
            return 50
        return (rsi_vals[-1] - mn) / (mx - mn) * 100

    def _vwap_dev(self, period=20):
        if len(self.closes) < period:
            return 0
        c = np.array(self.closes)[-period:]
        h = np.array(self.highs)[-period:]
        lo = np.array(self.lows)[-period:]
        v = np.array(self.volumes)[-period:]
        tp = (h + lo + c) / 3
        vwap = np.sum(tp * v) / np.sum(v) if np.sum(v) > 0 else c[-1]
        return (c[-1] - vwap) / vwap if vwap > 0 else 0

    def _bb(self, period=20, mult=2.0):
        if len(self.closes) < period:
            return 0, 0, 0
        c = np.array(self.closes)[-period:]
        ma = np.mean(c)
        std = np.std(c)
        return c[-1], ma + mult * std, ma - mult * std

    def _vol_ratio(self, period=20):
        if len(self.volumes) < period + 1:
            return 1.0
        v = np.array(self.volumes)
        avg = np.mean(v[-period - 1:-1])
        return v[-1] / avg if avg > 0 else 1.0

    def _atr(self, period=10):
        if len(self.closes) < period + 1:
            return 0
        h = np.array(self.highs)[-period - 1:]
        l = np.array(self.lows)[-period - 1:]
        c = np.array(self.closes)[-period - 1:]
        trs = []
        for i in range(1, len(h)):
            tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            trs.append(tr)
        return np.mean(trs) if trs else 0

    def _ema(self, period):
        if len(self.closes) < period:
            return 0
        c = np.array(self.closes)
        k = 2.0 / (period + 1)
        ema = np.mean(c[:period])
        for val in c[period:]:
            ema = val * k + ema * (1 - k)
        return ema

    def _pv_divergence(self):
        if len(self.closes) < 6:
            return 0, 0
        c = np.array(self.closes)[-6:]
        v = np.array(self.volumes)[-6:]
        pc = (c[-1] - c[0]) / c[0] if c[0] else 0
        vc = (v[-1] - v[0]) / v[0] if v[0] else 0
        if pc > 0.001 and vc < -0.1:
            return abs(pc), -1   # 价涨量缩 → 看跌背离
        if pc < -0.001 and vc < -0.1:
            return abs(pc), +1   # 价跌量缩 → 卖压减弱，偏多
        return 0, 0

    def _magnet(self):
        """
        [Fix12] 步长按价格数量级动态计算，不再硬编码1000/100
        例：BTC@95000 → step=1000；ETH@2500 → step=100；SOL@150 → step=10
        """
        if len(self.closes) == 0:
            return 0
        price = self.closes[-1]
        if price <= 0:
            return 0
        magnitude = 10 ** (len(str(int(price))) - 2)
        step = max(1, magnitude)
        lower = (price // step) * step
        upper = lower + step
        du = (upper - price) / price
        dd = (price - lower) / price
        if du < 0.003:
            return -1   # 接近上方整数关口 → 阻力
        if dd < 0.003:
            return +1   # 接近下方整数关口 → 支撑
        return 0

    def score(self):
        details = {}
        raw_score = 0

        # ── ATR波动率门槛：过滤低波动噪音单 ────────
        atr = self._atr(10)
        price = self.closes[-1] if self.closes else 0
        if price > 0 and atr * 10 < price * 0.0002:
            return 50, 0, {"ATR过滤": f"atr={atr:.2f} 波动率不足"}

        # ── RSI(14) ──────────────────────────────
        rsi = self._rsi(14)
        rsi_dir = 0
        if rsi < 20:
            rsi_dir = 1;  s = L1_WEIGHTS["RSI14"]
        elif rsi < 25:
            rsi_dir = 1;  s = L1_WEIGHTS["RSI14"] * 0.7
        elif rsi < 30:
            rsi_dir = 1;  s = L1_WEIGHTS["RSI14"] * 0.5
        elif rsi > 80:
            rsi_dir = -1; s = -L1_WEIGHTS["RSI14"]
        elif rsi > 75:
            rsi_dir = -1; s = -L1_WEIGHTS["RSI14"] * 0.7
        elif rsi > 70:
            rsi_dir = -1; s = -L1_WEIGHTS["RSI14"] * 0.5
        else:
            s = 0
        raw_score += s
        details["RSI14"] = f"{rsi:.1f}"

        # ── RSI + 放量 ───────────────────────────
        vr = self._vol_ratio()
        rvol_s = 0
        if vr > 1.0:
            if rsi < 25:
                rvol_s = L1_WEIGHTS["RSI_VOL"]
            elif rsi > 75:
                rvol_s = -L1_WEIGHTS["RSI_VOL"]
            elif rsi < 30:
                rvol_s = L1_WEIGHTS["RSI_VOL"] * 0.6
            elif rsi > 70:
                rvol_s = -L1_WEIGHTS["RSI_VOL"] * 0.6
        raw_score += rvol_s
        details["RSI+量"] = f"RSI{rsi:.0f} 量比{vr:.1f}x"

        # ── RSI + VWAP ───────────────────────────
        vd = self._vwap_dev()
        rv_s = 0
        if vd < -0.003 and rsi < 30:
            rv_s = L1_WEIGHTS["RSI_VWAP"]
        elif vd > 0.003 and rsi > 70:
            rv_s = -L1_WEIGHTS["RSI_VWAP"]
        raw_score += rv_s
        details["RSI+VWAP"] = f"偏离{vd * 100:.2f}%"

        # ── StochRSI ─────────────────────────────
        stoch = self._stoch_rsi()
        st_s = 0
        if stoch < 10:
            st_s = L1_WEIGHTS["STOCH_RSI"]
        elif stoch > 90:
            st_s = -L1_WEIGHTS["STOCH_RSI"]
        raw_score += st_s
        details["StochRSI"] = f"K={stoch:.0f}"

        # ── BB ───────────────────────────────────
        price, bb_up, bb_low = self._bb()
        bb_s = 0
        if price > 0:
            if price < bb_low:
                bb_s = L1_WEIGHTS["BB"]
            elif price > bb_up:
                bb_s = -L1_WEIGHTS["BB"]
        raw_score += bb_s
        details["BB"] = "下轨" if price < bb_low else ("上轨" if price > bb_up else "带内")

        # ── EMA共振 ──────────────────────────────
        ema3  = self._ema(3)
        ema7  = self._ema(7)
        ema21 = self._ema(21)
        ema55 = self._ema(55)
        ema_s = 0
        if ema55 > 0:
            bull = sum([price > ema3, ema3 > ema7, ema7 > ema21, ema21 > ema55])
            bear = sum([price < ema3, ema3 < ema7, ema7 < ema21, ema21 < ema55])
            if bull >= 3:
                ema_s = L1_WEIGHTS["EMA_RESON"] * bull / 4
            elif bear >= 3:
                ema_s = -L1_WEIGHTS["EMA_RESON"] * bear / 4
        raw_score += ema_s
        details["EMA共振"] = "多" if ema_s > 0 else ("空" if ema_s < 0 else "无")

        # ── 价量背离 ─────────────────────────────
        pv_val, pv_dir = self._pv_divergence()
        raw_score += pv_dir * L1_WEIGHTS["PV_DIV"]
        details["价量"] = "背离" if pv_dir != 0 else "正常"

        # ── 整数关口 ─────────────────────────────
        mag = self._magnet()
        raw_score += mag * L1_WEIGHTS["MAGNET"]
        details["关口"] = "近阻力" if mag < 0 else ("近支撑" if mag > 0 else "无")

        # ── VWAP偏离独立 ─────────────────────────
        vwap_s = 0
        if vd > 0.005:
            vwap_s = -L1_WEIGHTS["VWAP_DEV"]
        elif vd < -0.005:
            vwap_s = L1_WEIGHTS["VWAP_DEV"]
        raw_score += vwap_s
        # [Fix9] 补充独立 details 条目，方便排查
        details["VWAP偏离"] = f"{vd * 100:+.3f}% {'超买' if vwap_s < 0 else '超卖' if vwap_s > 0 else '正常'}"

        # ── 归一化 ───────────────────────────────
        normalized = (raw_score / L1_MAX) * 50 + 50

        # 带滞后的方向判断（防止边界跳动）
        prev = getattr(self, '_prev_dir', 0)
        if raw_score > 3:
            direction = 1
        elif raw_score < -3:
            direction = -1
        elif prev == 1 and raw_score > 0:
            direction = 1
        elif prev == -1 and raw_score < 0:
            direction = -1
        else:
            direction = 0
        self._prev_dir = direction

        return normalized, direction, details


# ───────────────────────────────────────────────
# 第二层：实时API数据
# ───────────────────────────────────────────────
class Layer2:
    def __init__(self, symbol):
        self.symbol = symbol
        # [Bug5 Fix] 正确提取base币种（去掉USDT后缀）
        self.base_asset = symbol.replace("USDT", "")
        self.spot_symbol = self.base_asset + "USDT"  # 对现货API而言相同，但语义清晰
        self.obi_history = deque(maxlen=120)
        self.taker_history = deque(maxlen=120)
        self.cvd_history = deque(maxlen=120)
        self.funding_rate = 0
        self.oi = 0
        self.oi_prev = 0
        # 缓存最近一次收盘价，用于OI方向判断
        self._last_close = 0
        self._prev_close = 0

    def collect(self):
        try:
            # ── 订单簿 ───────────────────────────
            ob = api_get(FAPI, "/fapi/v1/depth", {"symbol": self.symbol, "limit": 10})
            bids = [(float(p), float(q)) for p, q in ob["bids"]]
            asks = [(float(p), float(q)) for p, q in ob["asks"]]

            # OBI加权（前5档）
            w = [0.40, 0.25, 0.16, 0.11, 0.08]
            bv = sum(w[i] * bids[i][1] for i in range(min(5, len(bids))))
            av = sum(w[i] * asks[i][1] for i in range(min(5, len(asks))))
            tot = bv + av
            obi = (bv - av) / tot if tot else 0
            self.obi_history.append(obi)

            # 深度斜率
            bid_q = [q for _, q in bids[:5]]
            ask_q = [q for _, q in asks[:5]]
            x = np.arange(5, dtype=float)
            bid_slope = np.polyfit(x[:len(bid_q)], bid_q, 1)[0] if len(bid_q) >= 2 else 0
            ask_slope = np.polyfit(x[:len(ask_q)], ask_q, 1)[0] if len(ask_q) >= 2 else 0
            depth_slope = bid_slope - ask_slope

            # ── 最近成交（最近200条） ──
            trades = api_get(FAPI, "/fapi/v1/aggTrades", {
                "symbol": self.symbol,
                "limit": 200,
            })
            buy_vol = sum(float(t["q"]) for t in trades if not t["m"])
            sell_vol = sum(float(t["q"]) for t in trades if t["m"])
            total = buy_vol + sell_vol
            taker_ratio = buy_vol / total if total > 0 else 0.5
            self.taker_history.append(taker_ratio)

            # CVD（60秒窗口内累计净量）
            cvd = sum(float(t["q"]) * (1 if not t["m"] else -1) for t in trades)
            self.cvd_history.append(cvd)

            # 大单（用绝对量阈值避免稀疏数据失真）
            vols = [float(t["q"]) for t in trades]
            avg_v = np.mean(vols) if vols else 0
            threshold = avg_v * 5.0
            large_buy = sum(float(t["q"]) for t in trades
                            if not t["m"] and float(t["q"]) >= threshold)
            large_sell = sum(float(t["q"]) for t in trades
                             if t["m"] and float(t["q"]) >= threshold)

            # ── 资金费率 & 标记价格 ──────────────
            try:
                fr_data = api_get(FAPI, "/fapi/v1/premiumIndex", {"symbol": self.symbol})
                self.funding_rate = float(fr_data.get("lastFundingRate", 0))
                mark_price = float(fr_data.get("markPrice", 0))
                # 同步更新close缓存（用标记价格做OI方向判断）
                self._prev_close = self._last_close
                self._last_close = mark_price
            except Exception as e:
                logger.warning(f"[{self.symbol}] premiumIndex 请求失败: {e}")
                mark_price = 0

            # ── 未平仓量 OI ──────────────────────
            self.oi_prev = self.oi
            try:
                oi_data = api_get(FAPI, "/fapi/v1/openInterest", {"symbol": self.symbol})
                self.oi = float(oi_data.get("openInterest", 0))
            except Exception as e:
                logger.warning(f"[{self.symbol}] openInterest 请求失败: {e}")

            # ── 清算数据（暂不可用，allForceOrders返回400） ──
            long_liq = short_liq = 0

            # ── 永续溢价（合约价 vs 现货价） ──────
            # [Bug5 Fix] 语义上spot_symbol就是base+USDT，用SAPI访问现货
            try:
                spot = api_get(SAPI, "/api/v3/ticker/price", {"symbol": self.spot_symbol})
                spot_price = float(spot["price"])
                perp_prem = (
                    (mark_price - spot_price) / spot_price
                    if spot_price > 0 and mark_price > 0
                    else 0
                )
            except Exception as e:
                logger.warning(f"[{self.symbol}] spot ticker 请求失败: {e}")
                perp_prem = 0

            # 存储供score使用
            self._last = {
                "obi": obi,
                "depth_slope": depth_slope,
                "taker_ratio": taker_ratio,
                "cvd": cvd,
                "large_buy": large_buy,
                "large_sell": large_sell,
                "long_liq": long_liq,
                "short_liq": short_liq,
                "perp_prem": perp_prem,
            }
            return True
        except Exception as e:
            logger.warning(f"[{self.symbol}] Layer2.collect() 异常: {e}")
            return False

    def score(self):
        if not hasattr(self, "_last"):
            return 50, 0, {"状态": "数据不足"}

        d = self._last
        details = {}
        raw_score = 0

        # ── OBI ──────────────────────────────────
        obi = d["obi"]
        obi_s = 0
        if obi > 0.2:
            obi_s = L2_WEIGHTS["OBI"]
        elif obi > 0.1:
            obi_s = L2_WEIGHTS["OBI"] * 0.4
        elif obi < -0.2:
            obi_s = -L2_WEIGHTS["OBI"]
        elif obi < -0.1:
            obi_s = -L2_WEIGHTS["OBI"] * 0.4
        raw_score += obi_s
        details["OBI"] = f"{obi:+.3f} {'买厚' if obi > 0.1 else '卖厚' if obi < -0.1 else '均衡'}"

        # ── OBI持续 ──────────────────────────────
        persist_s = 0
        if len(self.obi_history) >= 6:
            recent = list(self.obi_history)[-6:]
            if all(v > 0.15 for v in recent):
                persist_s = L2_WEIGHTS["OBI_PERSIST"]
            elif all(v < -0.15 for v in recent):
                persist_s = -L2_WEIGHTS["OBI_PERSIST"]
        raw_score += persist_s
        details["OBI持续"] = "买>30s" if persist_s > 0 else ("卖>30s" if persist_s < 0 else "不持续")

        # ── OBI加速度 ────────────────────────────
        obi_spd_s = 0
        if len(self.obi_history) >= 10:
            arr = np.array(list(self.obi_history)[-10:])
            x = np.arange(len(arr))
            coeffs = np.polyfit(x, arr, 2)
            accel = coeffs[0] * 100
            speed = coeffs[1]
            if accel > 0.1 and speed > 0:
                obi_spd_s = L2_WEIGHTS["OBI_SPEED"]
            elif accel < -0.1 and speed < 0:
                obi_spd_s = -L2_WEIGHTS["OBI_SPEED"]
        raw_score += obi_spd_s
        details["OBI加速"] = "加速买" if obi_spd_s > 0 else ("加速卖" if obi_spd_s < 0 else "匀速")

        # ── 深度斜率 ─────────────────────────────
        ds = d["depth_slope"]
        ds_s = 0
        if ds > 0:
            ds_s = L2_WEIGHTS["DEPTH_SLOPE"] * min(1, ds / 100)
        elif ds < 0:
            ds_s = -L2_WEIGHTS["DEPTH_SLOPE"] * min(1, abs(ds) / 100)
        raw_score += ds_s
        details["深度斜率"] = f"{ds:+.1f}"

        # ── Taker ────────────────────────────────
        tr = d["taker_ratio"]
        tr_s = 0
        if tr > 0.60:
            tr_s = L2_WEIGHTS["TAKER"]
        elif tr > 0.55:
            tr_s = L2_WEIGHTS["TAKER"] * 0.3
        elif tr < 0.40:
            tr_s = -L2_WEIGHTS["TAKER"]
        elif tr < 0.45:
            tr_s = -L2_WEIGHTS["TAKER"] * 0.3
        raw_score += tr_s
        details["Taker"] = f"{tr:.3f} {'买多' if tr > 0.55 else '卖多' if tr < 0.45 else '均衡'}"

        # ── CVD趋势 ──────────────────────────────
        # [Fix7] 检查与取数的历史条数统一，用实际可用数量
        cvd_s = 0
        cvd_hist_len = len(self.cvd_history)
        if cvd_hist_len >= 5:
            take_n = min(10, cvd_hist_len)
            arr = np.array(list(self.cvd_history)[-take_n:])
            x = np.arange(len(arr))
            slope = np.polyfit(x, arr, 1)[0]
            if slope > 0:
                cvd_s = L2_WEIGHTS["CVD"] * min(1, slope / 10)
            elif slope < 0:
                cvd_s = -L2_WEIGHTS["CVD"] * min(1, abs(slope) / 10)
        raw_score += cvd_s
        details["CVD"] = "上升" if cvd_s > 0 else ("下降" if cvd_s < 0 else "平")

        # ── 大单 ─────────────────────────────────
        lb, ls = d["large_buy"], d["large_sell"]
        lo_s = 0
        total_large = lb + ls
        if total_large > 0:
            ratio = (lb - ls) / total_large
            if ratio > 0.3:
                lo_s = L2_WEIGHTS["LARGE_ORDER"]
            elif ratio < -0.3:
                lo_s = -L2_WEIGHTS["LARGE_ORDER"]
        raw_score += lo_s
        details["大单"] = f"买{lb:.0f}/卖{ls:.0f}"

        # ── 清算 ─────────────────────────────────
        llq, slq = d["long_liq"], d["short_liq"]
        liq_s = 0
        if llq + slq > 50000:
            ratio = (slq - llq) / (llq + slq)
            if ratio > 0.3:
                liq_s = L2_WEIGHTS["LIQUIDATION"]
            elif ratio < -0.3:
                liq_s = -L2_WEIGHTS["LIQUIDATION"]
        raw_score += liq_s
        details["清算"] = f"多爆{llq / 1000:.0f}k/空爆{slq / 1000:.0f}k"

        # ── 资金费率 ─────────────────────────────
        fr = self.funding_rate
        fr_s = 0
        if fr > 0.0005:
            fr_s = -L2_WEIGHTS["FUNDING"]
        elif fr < -0.0005:
            fr_s = L2_WEIGHTS["FUNDING"]
        raw_score += fr_s
        details["费率"] = f"{fr * 100:.4f}%"

        # ── OI变化 ───────────────────────────────
        # [Bug3 Fix] 原代码oi_s永远赋值0。
        # 正确逻辑：OI增加配合价格方向判断趋势，OI减少=平仓无方向
        oi_s = 0
        if self.oi_prev > 0 and self.oi > 0:
            chg = (self.oi - self.oi_prev) / self.oi_prev
            if chg > 0.001:
                # OI增加时，结合价格方向判断多空
                if self._last_close > 0 and self._prev_close > 0:
                    price_up = self._last_close > self._prev_close
                    # 价涨+OI增 → 趋势延续看多；价跌+OI增 → 趋势延续看空
                    oi_s = L2_WEIGHTS["OI_CHANGE"] if price_up else -L2_WEIGHTS["OI_CHANGE"]
            elif chg < -0.001:
                # OI减少 = 平仓，方向不明，不加分
                oi_s = 0
            details["OI"] = f"{chg * 100:+.3f}%"
        else:
            details["OI"] = "N/A"
        raw_score += oi_s

        # ── 永续溢价 ─────────────────────────────
        pp = d["perp_prem"]
        pp_s = 0
        if pp > 0.0005:
            pp_s = -L2_WEIGHTS["PERP_PREM"]
        elif pp < -0.0005:
            pp_s = L2_WEIGHTS["PERP_PREM"]
        raw_score += pp_s
        details["溢价"] = f"{pp * 100:+.4f}%"

        normalized = (raw_score / L2_MAX) * 50 + 50

        # 带滞后的方向判断
        prev = getattr(self, '_prev_dir', 0)
        if raw_score > 3:
            direction = 1
        elif raw_score < -3:
            direction = -1
        elif prev == 1 and raw_score > 0:
            direction = 1
        elif prev == -1 and raw_score < 0:
            direction = -1
        else:
            direction = 0
        self._prev_dir = direction

        return normalized, direction, details


# ───────────────────────────────────────────────
# 信号历史 & 胜率统计
# ───────────────────────────────────────────────
class SignalHistory:
    def __init__(self, filepath):
        self.filepath = filepath
        # [Fix10] 加锁防止并发读写竞争
        self._lock = threading.Lock()
        self.records = self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"历史记录加载失败: {e}")
        return []

    def _save(self):
        # 调用方已持有锁
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.records[-5000:], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"历史记录保存失败: {e}")

    def add(self, symbol, direction, score, price, details, conf_num=0):
        with self._lock:
            self.records.append({
                "ts": int(time.time() * 1000),
                "sym": symbol,
                "dir": direction,
                "score": score,
                "conf": conf_num,
                "price": price,
                "details": details,
                "verified": False,
                "result": None,  # 1=赢, 0=输
            })
            self._save()

    def verify(self, symbol, current_price):
        """验证历史信号"""
        now = int(time.time() * 1000)
        settle_ms = SETTLE_MINS * 60 * 1000
        updated = False
        with self._lock:
            for rec in self.records:
                if rec["verified"] or rec["sym"] != symbol:
                    continue
                if now - rec["ts"] < settle_ms:
                    continue
                if rec["dir"] == 1:
                    rec["result"] = 1 if current_price > rec["price"] else 0
                else:
                    rec["result"] = 1 if current_price < rec["price"] else 0
                rec["verified"] = True
                rec["settle_price"] = current_price
                updated = True
            if updated:
                self._save()

    def stats(self, symbol=None, last_n=100):
        """统计胜率"""
        with self._lock:
            recs = [r for r in self.records if r["sym"] == symbol] if symbol else self.records
            verified = [r for r in recs if r["verified"]]
            unsettled = len(recs) - len(verified)
            if not verified:
                return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "unsettled": unsettled, "total_all": 0}
            recent = verified[-last_n:]
            total = len(recent)
            wins = sum(1 for r in recent if r["result"] == 1)
            return {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": wins / total * 100 if total > 0 else 0,
                "unsettled": unsettled,
                "total_all": len(verified),
            }

    def last_record_for(self, symbol):
        """返回指定品种最新一条记录（线程安全）"""
        with self._lock:
            for rec in reversed(self.records):
                if rec["sym"] == symbol:
                    return rec
        return None


# ───────────────────────────────────────────────
# 第三层：5分钟K线趋势（方案B）
# ───────────────────────────────────────────────
class Layer3:
    def __init__(self, symbol, lookback=50):
        self.symbol = symbol
        self.lookback = lookback
        self.closes = deque(maxlen=lookback)
        self.highs = deque(maxlen=lookback)
        self.lows = deque(maxlen=lookback)
        self.last_bar_ts = 0
        self._init_history()

    def _init_history(self):
        resp = session.get(f"{FAPI}/fapi/v1/klines", params={
            "symbol": self.symbol, "interval": "5m", "limit": self.lookback
        }, timeout=10)
        resp.raise_for_status()
        for k in resp.json():
            self.closes.append(float(k[4]))
            self.highs.append(float(k[2]))
            self.lows.append(float(k[3]))
            self.last_bar_ts = int(k[0])

    def update(self):
        try:
            resp = session.get(f"{FAPI}/fapi/v1/klines", params={
                "symbol": self.symbol, "interval": "5m", "limit": 2
            }, timeout=3)
            resp.raise_for_status()
            klines = resp.json()
            if len(klines) >= 2:
                k = klines[-2]
                bar_ts = int(k[0])
                if bar_ts > self.last_bar_ts:
                    self.closes.append(float(k[4]))
                    self.highs.append(float(k[2]))
                    self.lows.append(float(k[3]))
                    self.last_bar_ts = bar_ts
        except Exception as e:
            logger.warning(f"[{self.symbol}] 5m K线更新失败: {e}")

    def _ema(self, period):
        if len(self.closes) < period:
            return 0
        c = np.array(self.closes)
        k = 2.0 / (period + 1)
        ema = np.mean(c[:period])
        for val in c[period:]:
            ema = val * k + ema * (1 - k)
        return ema

    def _rsi(self, period=14):
        if len(self.closes) < period + 1:
            return 50
        c = np.array(self.closes)
        deltas = np.diff(c[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def _bb(self, period=20, mult=2.0):
        if len(self.closes) < period:
            return 0, 0, 0
        c = np.array(self.closes)[-period:]
        ma = np.mean(c)
        std = np.std(c)
        return c[-1], ma + mult * std, ma - mult * std

    def score(self):
        details = {}
        price = self.closes[-1] if self.closes else 0

        # EMA趋势方向
        ema7 = self._ema(7)
        ema21 = self._ema(21)
        ema_dir = 0
        if ema7 > 0 and ema21 > 0:
            if ema7 > ema21 * 1.001:
                ema_dir = 1
            elif ema7 < ema21 * 0.999:
                ema_dir = -1
        details["5m_EMA"] = "多" if ema_dir > 0 else ("空" if ema_dir < 0 else "平")

        # RSI
        rsi = self._rsi(14)
        rsi_dir = 0
        if rsi < 35:
            rsi_dir = 1
        elif rsi > 65:
            rsi_dir = -1
        details["5m_RSI"] = f"{rsi:.0f}"

        # BB位置
        p, bb_up, bb_low = self._bb()
        bb_dir = 0
        if p > 0:
            if p < bb_low:
                bb_dir = 1
            elif p > bb_up:
                bb_dir = -1
        details["5m_BB"] = "下轨" if bb_dir > 0 else ("上轨" if bb_dir < 0 else "带内")

        # 综合5m方向：EMA + RSI + BB
        votes = ema_dir + rsi_dir + bb_dir
        if votes >= 2:
            direction = 1
        elif votes <= -2:
            direction = -1
        else:
            direction = 0

        return direction, details


# ───────────────────────────────────────────────
# 综合引擎
# ───────────────────────────────────────────────
class SignalEngine:
    def __init__(self, symbol_key):
        self.key = symbol_key
        self.symbol = SYMBOLS[symbol_key]
        self.layer1 = Layer1(self.symbol)
        self.layer2 = Layer2(self.symbol)
        self.layer3 = Layer3(self.symbol)

    def update_kline(self, close, high, low, open_p, volume, taker_buy):
        self.layer1.update(close, high, low, open_p, volume, taker_buy)

    def collect(self):
        return self.layer2.collect()

    def update_5m(self):
        self.layer3.update()

    def evaluate(self):
        l1s, l1d, l1det = self.layer1.score()
        l2s, l2d, l2det = self.layer2.score()
        l3d, l3det = self.layer3.score()

        total = l1s * 0.62 + l2s * 0.38  # L1权重略高

        # L1/L2一致性加成
        if l1d == l2d and l1d != 0:
            total += 5 * l1d

        # Layer3(5m趋势)过滤：方向冲突时降分
        l3_label = "顺"
        if l3d != 0:
            l1l2_dir = 1 if total > 50 else (-1 if total < 50 else 0)
            if l1l2_dir != 0 and l3d != l1l2_dir:
                total += -8 * l1l2_dir  # 5m反向，惩罚8分
                l3_label = "逆"
            elif l1l2_dir != 0 and l3d == l1l2_dir:
                total += 3 * l3d  # 5m同向，奖励3分

        total = max(0, min(100, total))

        # 信号强度：离中性区(50)越远越强
        strength = abs(total - 50)

        # 方向
        if total > 55:
            direction = 1
        elif total < 45:
            direction = -1
        else:
            direction = 0

        # ── 5m_RSI逆势过滤 ──────────────────────
        # 看空+5m_RSI<50 → 顺势(5m已超卖) → 拒绝
        # 看多+5m_RSI>50 → 顺势(5m已超买) → 拒绝
        rsi_5m = l3det.get("5m_RSI", "50")
        try:
            rsi_5m_val = float(rsi_5m)
        except (ValueError, TypeError):
            rsi_5m_val = 50
        if direction == -1 and rsi_5m_val < 50:
            direction = 0  # 看空但5m已偏弱，拒绝
        elif direction == 1 and rsi_5m_val > 50:
            direction = 0  # 看多但5m已偏强，拒绝

        # ── 置信度 ───────────────────────────────
        # [Bug2 Fix] 按agree_label分别选取对应的base/mult常量
        if l1d == l2d and l1d != 0:
            agree = 1.0
            agree_label = "一致"
            base, mult = CONF_AGREE_BASE, CONF_AGREE_MULT
        elif l1d != 0 and l2d != 0 and l1d != l2d:
            agree = 0.3
            agree_label = "冲突"
            base, mult = CONF_CONFLICT_BASE, CONF_CONFLICT_MULT
        elif l1d != 0:
            agree = 0.6
            agree_label = "仅L1"
            base, mult = CONF_ONLYL1_BASE, CONF_ONLYL1_MULT
        elif l2d != 0:
            agree = 0.45
            agree_label = "仅L2"
            base, mult = CONF_ONLYL2_BASE, CONF_ONLYL2_MULT
        else:
            agree = 0.2
            agree_label = "无"
            base, mult = CONF_NONE_BASE, CONF_NONE_MULT

        conf_num = agree * (base + strength * mult)
        conf_num = max(0, min(100, conf_num))

        # 文字标签由数值决定
        if conf_num >= 65:
            conf = "高"
        elif conf_num >= 45:
            conf = "中"
        else:
            conf = "低"

        return {
            "direction": direction,
            "dir_str": "看涨 ↑" if direction == 1 else ("看跌 ↓" if direction == -1 else "观望 —"),
            "total": total,
            "confidence": conf,
            "conf_num": round(conf_num, 1),
            "agree": agree_label,
            "l1_score": l1s,
            "l2_score": l2s,
            "l3_dir": l3d,
            "l3_label": l3_label,
            "l1_details": l1det,
            "l2_details": l2det,
            "l3_details": l3det,
        }


# ───────────────────────────────────────────────
# 主程序
# ───────────────────────────────────────────────
def main():
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ["BTC", "ETH"]
    valid = [s.upper() for s in symbols if s.upper() in SYMBOLS]
    if not valid:
        valid = ["BTC", "ETH"]

    history = SignalHistory(HISTORY_FILE)
    engines = {k: SignalEngine(k) for k in valid}
    last_state = {}  # 跟踪上次信号状态，只在变化时显示

    print(f"\n{'=' * 65}")
    print(f"事件合约信号系统 v5.3 | {', '.join(valid)}")
    print(f"第一层(1m): RSI+放量+VWAP+BB+EMA共振+关口")
    print(f"第二层(实时): OBI+持续+加速+深度斜率+Taker+CVD+大单+费率+OI+溢价")
    print(f"第三层(5m): EMA趋势+RSI+BB（方向过滤）")
    print(f"配置: MIN_CONF={MIN_CONF}, ATR门槛, BB mult=2.0, RSI_VOL vr>1.0")
    print(f"信号历史: {history.filepath}")
    print(f"{'=' * 65}\n")

    round_count = 0
    try:
        while True:
            round_count += 1

            def collect_one(key, engine):
                """采集单个品种数据"""
                # 更新1m K线（取倒数第二根已收盘K线）
                try:
                    resp = session.get(f"{FAPI}/fapi/v1/klines", params={
                        "symbol": engine.symbol, "interval": "1m", "limit": 2
                    }, timeout=3)
                    resp.raise_for_status()
                    klines = resp.json()
                    if len(klines) >= 2:
                        k = klines[-2]
                        engine.update_kline(
                            float(k[4]), float(k[2]), float(k[3]),
                            float(k[1]), float(k[5]), float(k[9])
                        )
                except Exception as e:
                    logger.warning(f"[{key}] K线更新失败: {e}")

                # 更新5m K线（Layer3）
                engine.update_5m()

                # 采集实时数据
                engine.collect()

                # 验证历史信号
                try:
                    price_resp = session.get(f"{FAPI}/fapi/v1/ticker/price", params={
                        "symbol": engine.symbol
                    }, timeout=3)
                    price_resp.raise_for_status()
                    current_price = float(price_resp.json()["price"])
                    history.verify(engine.symbol, current_price)
                except Exception as e:
                    logger.warning(f"[{key}] 价格验证失败: {e}")

            # 并发采集BTC+ETH
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = {ex.submit(collect_one, k, e): k for k, e in engines.items()}
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception as e:
                        logger.warning(f"collect_one 异常: {e}")

            # 评估 & 记录
            has_activity = False
            for key, engine in engines.items():
                result = engine.evaluate()
                price = engine.layer1.closes[-1] if engine.layer1.closes else 0

                # 置信度过滤：低于阈值视为无信号
                if result["conf_num"] < MIN_CONF:
                    result["direction"] = 0
                    result["dir_str"] = "观望 —"

                # 记录有方向的信号
                # [Fix6] 去重：同品种同方向 5分钟冷却；反向信号直接记录
                recorded = False
                if result["direction"] != 0:
                    should_record = True
                    last_rec = history.last_record_for(engine.symbol)
                    if last_rec is not None:
                        elapsed_ms = int(time.time() * 1000) - last_rec["ts"]
                        if last_rec["dir"] == result["direction"] and elapsed_ms < 5 * 60 * 1000:
                            should_record = False

                    if should_record:
                        det_str = "; ".join(
                            f"{k}={v}" for k, v in result["l1_details"].items()
                        )
                        l3_str = "; ".join(
                            f"{k}={v}" for k, v in result["l3_details"].items()
                        )
                        det_str += "; " + l3_str
                        history.add(
                            engine.symbol, result["direction"],
                            result["total"], price, det_str, result["conf_num"]
                        )
                        recorded = True

                # 只在信号方向变化或新记录时显示
                state_key = f"{key}"
                prev = last_state.get(state_key, {})
                changed = (prev.get("dir") != result["direction"] or
                          prev.get("conf_tier") != result["confidence"])
                last_state[state_key] = {"dir": result["direction"], "conf_tier": result["confidence"]}

                if changed or recorded:
                    has_activity = True
                    now = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M:%S")
                    l3_str = "↑" if result['l3_dir'] > 0 else ("↓" if result['l3_dir'] < 0 else "—")
                    print(f"\n  [{now}] [{key}] {result['dir_str']}  "
                          f"置信:{result['confidence']}({result['conf_num']})[{result['agree']}]  "
                          f"综合:{result['total']:.1f}  "
                          f"(L1={result['l1_score']:.1f} L2={result['l2_score']:.1f} "
                          f"5m:{l3_str}{result['l3_label']})  "
                          f"价格:{price:.2f}")
                    if recorded:
                        print(f"    >>> 已记录信号")

            # 胜率统计（有活动时显示）
            if has_activity:
                for key, engine in engines.items():
                    stats = history.stats(engine.symbol, 50)
                    if stats and stats["total"] >= 5:
                        print(
                            f"  [{key}] 近{stats['total']}次信号胜率: {stats['win_rate']:.1f}% "
                            f"({stats['wins']}赢/{stats['losses']}输/"
                            f"{stats['total']}已结算/{stats['unsettled']}未结算)"
                            f"总信号:{stats['total_all']}"
                        )

            time.sleep(COLLECT_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n{'=' * 65}")
        print("已停止 | 信号历史胜率:")
        for key, engine in engines.items():
            stats = history.stats(engine.symbol, 200)
            if stats:
                print(
                    f"  {key}: {stats['win_rate']:.1f}% "
                    f"({stats['wins']}赢/{stats['losses']}输/"
                    f"{stats['total']}已结算/{stats['unsettled']}未结算)"
                    f"总信号:{stats['total_all']}"
                )
        print(f"{'=' * 65}")


if __name__ == "__main__":
    main()