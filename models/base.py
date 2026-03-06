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

SEED      = 42
CACHE_DIR = Path("/tmp/p2_model_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Clear any v1 cache files (missing max_daily_date field)
for _f in CACHE_DIR.glob("*.pkl"):
	try:
		import pickle as _pkl
		with open(_f, "rb") as _fh:
			_d = _pkl.load(_fh)
		# If any result dict lacks max_daily_date, bust the whole cache
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
	"""
	Build sequences for time series modeling.
	"""
	n_samples = len(features)
	if n_samples <= lookback:
		raise ValueError(
			f"Insufficient data for lookback={lookback}. "
			f"Need at least {lookback + 1} samples, but got {n_samples}. "
			f"Try selecting an earlier start year or smaller lookback."
		)
	
	X, y = [], []
	for i in range(lookback, n_samples):
		X.append(features[i - lookback: i])
		y.append(targets[i])
	
	X_arr = np.array(X, dtype=np.float32)
	y_arr = np.array(y, dtype=np.float32)
	
	if X_arr.ndim != 3:
		raise ValueError(f"Expected X to be 3D (samples, lookback, features), got shape {X_arr.shape}")
	
	return X_arr, y_arr


# ── Train / val / test split ──────────────────────────────────────────────────

def train_val_test_split(X, y, train_pct=0.70, val_pct=0.15):
	"""
	Split data into train/val/test sets.
	"""
	n = len(X)
	if n == 0:
		raise ValueError("Cannot split empty array.")
	
	t1 = int(n * train_pct)
	t2 = int(n * (train_pct + val_pct))
	
	min_size = 1
	if t1 < min_size:
		raise ValueError(f"Training set too small: {t1} samples.")
	if t2 - t1 < min_size:
		raise ValueError(f"Validation set too small: {t2 - t1} samples.")
	if n - t2 < min_size:
		raise ValueError(f"Test set too small: {n - t2} samples.")
	
	return X[:t1], y[:t1], X[t1:t2], y[t1:t2], X[t2:], y[t2:]


# ── Feature scaling ───────────────────────────────────────────────────────────

def scale_features(X_train, X_val, X_test):
	"""
	Scale features using RobustScaler.
	"""
	if X_train.size == 0:
		raise ValueError("X_train is empty.")
	
	if X_train.ndim == 2:
		X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
		if X_val.size > 0:
			X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
		if X_test.size > 0:
			X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
	elif X_train.ndim != 3:
		raise ValueError(f"Expected 2D or 3D X_train, got shape {X_train.shape}")
	
	n_feat = X_train.shape[2]
	scaler = RobustScaler()
	scaler.fit(X_train.reshape(-1, n_feat))
	
	def _t(X):
		if X.size == 0:
			return X
		s = X.shape
		return scaler.transform(X.reshape(-1, n_feat)).reshape(s)
	
	return _t(X_train), _t(X_val), _t(X_test), scaler


# ── Label builder ─────────────────────────────────────────────────────────────

def returns_to_labels(y_raw):
	"""
	Convert returns/probabilities to integer labels.
	"""
	y_raw = np.asarray(y_raw)
	
	if y_raw.ndim == 1:
		return y_raw.astype(np.int32)
	elif y_raw.ndim == 2:
		return np.argmax(y_raw, axis=1).astype(np.int32)
	else:
		raise ValueError(f"Expected 1D or 2D array, got shape {y_raw.shape}")


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(y_labels, n_classes):
	"""
	Compute balanced class weights.
	"""
	if len(y_labels) == 0:
		return {c: 1.0 for c in range(n_classes)}
	
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
	Find optimal lookback by testing candidates.
	"""
	from tensorflow import keras

	if candidates is None:
		candidates = [30, 45, 60]

	best_lb, best_loss = candidates[0], np.inf

	for lb in candidates:
		try:
			X_seq, y_seq = build_sequences(X_raw, y_raw, lb)
			y_lab        = returns_to_labels(y_seq)
			X_tr, y_tr, X_v, y_v, _, _ = train_val_test_split(X_seq, y_lab, train_pct, val_pct)
			X_tr_s, X_v_s, _, _        = scale_features(X_tr, X_v, X_v)
			cw = compute_class_weights(y_tr, n_classes)

			inp = keras.Input(shape=X_tr_s.shape[1:])
			x   = keras.layers.Conv1D(16, min(3, lb), padding="causal", activation="relu")(inp)
			x   = keras.layers.GlobalAveragePooling1D()(x)
			out = keras.layers.Dense(n_classes, activation="softmax")(x)
			m   = keras.Model(inp, out)
			m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

			hist = m.fit(
				X_tr_s, y_tr,
				validation_data=(X_v_s, y_v),
				epochs=15, batch_size=64, class_weight=cw,
				callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
				verbose=0,
			)
			val_loss = min(hist.history.get("val_loss", [np.inf]))
			if val_loss < best_loss:
				best_loss, best_lb = val_loss, lb
			del m
		except Exception:
			continue

	return best_lb
