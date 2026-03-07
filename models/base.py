"""
models/base.py
Shared utilities for all CNN-LSTM variants.
Optimised for CPU training on HF Spaces.
"""

import numpy as np
import hashlib
import pickle
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
CACHE_DIR = Path("/tmp/p2_model_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Clear any v1 cache files (missing max_daily_date field)
for _f in CACHE_DIR.glob("*.pkl"):
	try:
		import pickle as _pkl
		with open(_f, "rb") as _fh:
			_d = _pkl.load(_fh)
		if isinstance(_d, dict) and "results" in _d:
			_needs_bust = any(
				isinstance(r, dict) and "max_daily_date" not in r
				for r in _d["results"].values() if r is not None
			)
			if _needs_bust:
				_f.unlink(missing_ok=True)
	except Exception:
		_f.unlink(missing_ok=True)

np.random.seed(SEED)

# Minimum training samples needed — anything below this is not worth fitting
MIN_TRAIN_SAMPLES = 30


# ── Cache helpers ─────────────────────────────────────────────────────────────

def make_cache_key(last_date, start_yr, fee_bps, epochs, split, include_cash, lookback):
	raw = f"v2_{last_date}_{start_yr}_{fee_bps}_{epochs}_{split}_{include_cash}_{lookback}"
	return hashlib.md5(raw.encode()).hexdigest()


def save_cache(key, payload):
	with open(CACHE_DIR / f"{key}.pkl", "wb") as f:
		pickle.dump(payload, f)


def load_cache(key):
	path = CACHE_DIR / f"{key}.pkl"
	if path.exists():
		try:
			with open(path, "rb") as f:
				return pickle.load(f)
		except Exception:
			path.unlink(missing_ok=True)
	return None


# ── Sequence builder ──────────────────────────────────────────────────────────

def build_sequences(features, targets, lookback):
	X, y = [], []
	for i in range(lookback, len(features)):
		X.append(features[i - lookback: i])
		y.append(targets[i])
	return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Train / val / test split ──────────────────────────────────────────────────

def train_val_test_split(X, y, train_pct=0.70, val_pct=0.15):
	n = len(X)
	t1 = int(n * train_pct)
	t2 = int(n * (train_pct + val_pct))
	return X[:t1], y[:t1], X[t1:t2], y[t1:t2], X[t2:], y[t2:]


# ── Feature scaling ───────────────────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
	"""
	Fit RobustScaler on X_train and transform all three splits.

	FIX: Added explicit empty-array guard before the reshape call.
	The original code called X_train.reshape(-1, n_feat) with no check,
	so when X_train had size=0 (start_yr too recent → too few rows after
	lookback + split), numpy raised:
	    ValueError: cannot reshape array of size 0 into shape (0,n_feat)
	Now raises a clear, actionable ValueError instead.
	"""
	# ── Guard: empty or too-small training set ────────────────────────────────
	if X_train.size == 0 or len(X_train) == 0:
		raise ValueError(
			f"X_train is empty (shape={X_train.shape}). "
			f"The chosen start_year leaves too few rows after feature engineering "
			f"and the lookback window. Try an earlier start year."
		)
	if len(X_train) < MIN_TRAIN_SAMPLES:
		raise ValueError(
			f"X_train has only {len(X_train)} samples (minimum={MIN_TRAIN_SAMPLES}). "
			f"Try an earlier start year or reduce the lookback window."
		)

	# Ensure at least 2D
	if X_train.ndim == 1:
		X_train = X_train.reshape(1, -1)

	# Handle 2D: (samples, features) → (samples, 1, features)
	if X_train.ndim == 2:
		X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
		if X_val.size > 0:
			X_val = (X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
					 if X_val.ndim == 2 else X_val.reshape(1, 1, -1))
		if X_test.size > 0:
			X_test = (X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
					  if X_test.ndim == 2 else X_test.reshape(1, 1, -1))

	if X_train.ndim != 3:
		raise ValueError(
			f"X_train must be 3D after reshaping, got shape {X_train.shape}"
		)

	n_feat = X_train.shape[2]
	scaler = RobustScaler()
	scaler.fit(X_train.reshape(-1, n_feat))

	def _t(X):
		if X.size == 0:
			return X
		s = X.shape
		return scaler.transform(X.reshape(-1, n_feat)).reshape(s)

	return _t(X_train), _t(X_val), _t(X_test), scaler


# ── Label builder (no CASH class — CASH is a risk overlay) ───────────────────

def returns_to_labels(y_raw):
	"""Simple argmax — model always predicts one of the ETFs."""
	y_raw = np.asarray(y_raw)
	if y_raw.ndim == 1:
		return y_raw.astype(np.int32)
	return np.argmax(y_raw, axis=1).astype(np.int32)


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(y_labels, n_classes):
	present = np.unique(y_labels)
	try:
		weights = compute_class_weight("balanced", classes=present, y=y_labels)
		weight_dict = {int(c): float(w) for c, w in zip(present, weights)}
	except Exception:
		weight_dict = {}
	for c in range(n_classes):
		if c not in weight_dict:
			weight_dict[c] = 1.0
	return weight_dict


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(patience_es=15, patience_lr=8, min_lr=1e-6):
	from tensorflow import keras
	return [
		keras.callbacks.EarlyStopping(
			monitor="val_loss", patience=patience_es,
			restore_best_weights=True, verbose=0,
		),
		keras.callbacks.ReduceLROnPlateau(
			monitor="val_loss", factor=0.5,
			patience=patience_lr, min_lr=min_lr, verbose=0,
		),
	]


# ── Output head ───────────────────────────────────────────────────────────────

def classification_head(x, n_classes, dropout=0.3):
	from tensorflow import keras
	x = keras.layers.Dense(32, activation="relu")(x)
	x = keras.layers.Dropout(dropout)(x)
	x = keras.layers.Dense(n_classes, activation="softmax")(x)
	return x


# ── Auto lookback selection ───────────────────────────────────────────────────

def find_best_lookback(X_raw, y_raw, train_pct, val_pct, n_classes,
					   include_cash=False, candidates=None):
	"""
	Try each lookback candidate and return the one with the lowest val loss.

	FIX vs original:
	  1. Validates X_train is non-empty BEFORE calling scale_features.
	     Original silently caught all exceptions and returned candidates[0]=30
	     even when 30d also failed — causing the main run to crash with the
	     same cryptic reshape error.
	  2. If ALL candidates fail, raises ValueError with a clear message
	     (app.py catches this and shows st.error with actionable advice).
	"""
	from tensorflow import keras

	if candidates is None:
		candidates = [30, 45, 60]

	best_lb, best_loss = None, np.inf
	skip_reasons = {}

	for lb in candidates:
		try:
			X_seq, y_seq = build_sequences(X_raw, y_raw, lb)
			y_lab = returns_to_labels(y_seq)
			X_tr, y_tr, X_v, y_v, X_te, _ = train_val_test_split(
				X_seq, y_lab, train_pct, val_pct
			)

			# Guard: skip if splits are too small
			if len(X_tr) < MIN_TRAIN_SAMPLES:
				skip_reasons[lb] = f"X_train too small ({len(X_tr)} < {MIN_TRAIN_SAMPLES})"
				continue
			if len(X_v) == 0:
				skip_reasons[lb] = "X_val is empty"
				continue

			X_tr_s, X_v_s, _, _ = scale_features(X_tr, X_v, X_te)
			cw = compute_class_weights(y_tr, n_classes)

			inp = keras.Input(shape=X_tr_s.shape[1:])
			x = keras.layers.Conv1D(
				16, min(3, lb), padding="causal", activation="relu"
			)(inp)
			x = keras.layers.GlobalAveragePooling1D()(x)
			out = keras.layers.Dense(n_classes, activation="softmax")(x)
			m = keras.Model(inp, out)
			m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

			hist = m.fit(
				X_tr_s, y_tr,
				validation_data=(X_v_s, y_v),
				epochs=15, batch_size=64, class_weight=cw,
				callbacks=[keras.callbacks.EarlyStopping(
					patience=5, restore_best_weights=True
				)],
				verbose=0,
			)
			val_loss = min(hist.history.get("val_loss", [np.inf]))
			if val_loss < best_loss:
				best_loss, best_lb = val_loss, lb
			del m

		except Exception as e:
			skip_reasons[lb] = str(e)
			continue

	if best_lb is None:
		reasons_str = "; ".join(f"lb={k}: {v}" for k, v in skip_reasons.items())
		raise ValueError(
			f"All lookback candidates {candidates} failed. "
			f"The chosen start_year likely leaves too few data rows. "
			f"Details — {reasons_str}"
		)

	return best_lb
