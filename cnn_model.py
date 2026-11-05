"""
cnn_model.py — Stage B, Person 4: small CNN for the Spurious-Feature project.

Plugs into the frozen shared core WITHOUT changing any score function:
  * takes FLAT inputs (N, 3*28*28) like every other model, reshapes inside
  * exposes fit / predict / predict_proba / classes_ (sklearn-style)
  * pickles to CPU tensors, so the OOD/merge notebook can load it with or
    without a GPU (the merge notebook must be able to `import cnn_model`)
  * deterministic given `seed` (same seed -> same weights on the same device)
  * early stopping uses an internal split of the TRAINING data only (E1),
    never the scoring/validation set and never any OOD data

Usage in the CNN notebook (after `from core import *`):
    import cnn_model
    cnn_model.register(MODEL_REGISTRY, REG_LEVELS)
"""
import os
import copy
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # needed for deterministic cuBLAS
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_SHAPE = (3, 28, 28)

# Regularisation levels = Adam L2 weight decay (parallels the MLP's `alpha`).
CNN_REG_LEVELS = {"low": 0.0, "default": 1e-4, "high": 1e-2}

# Hyperparameters. Fix these on E1 validation ONLY, before anyone looks at OOD numbers,
# then do not change them (they are written into the frozen manifest via CONFIG).
CNN_HPARAMS = {
    "channels": (16, 32),     # conv widths
    "hidden": 64,             # fully connected width
    "lr": 3e-3,               # chosen on E1 validation (tuning_e1_only.csv)
    "batch_size": 128,
    "max_epochs": 15,
    "patience": 3,            # early stopping on the internal E1 split
    "inner_val_frac": 0.1,
}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallCNN(nn.Module):
    """2 conv blocks + 1 hidden FC layer, ~110k parameters.
    'same' padding keeps the 3x3 corner markers visible to the first conv layer."""

    def __init__(self, channels=(16, 32), hidden=64, n_classes=2):
        super().__init__()
        c1, c2 = channels
        self.conv1 = nn.Conv2d(3, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc1 = nn.Linear(c2 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14 -> 7
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


class CNNClassifier:
    """sklearn-style wrapper so the shared core can call model.predict(flat(X))."""

    def __init__(self, weight_decay=1e-4, seed=0, **hparams):
        self.weight_decay = weight_decay
        self.seed = seed
        self.hp = {**CNN_HPARAMS, **hparams}
        self.classes_ = np.array([0, 1])
        self.net = None
        self.history_ = []          # per-epoch (train_loss, inner_val_acc)
        self.best_epoch_ = None

    # ---------- helpers ----------
    @staticmethod
    def _to_img(X):
        X = np.asarray(X, dtype=np.float32)
        return X.reshape(len(X), *IMG_SHAPE)

    def _seed_everything(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    # ---------- sklearn API ----------
    def fit(self, X, y):
        self._seed_everything()
        dev = _device()
        X = self._to_img(X)
        y = np.asarray(y, dtype=np.int64)

        # internal split of the E1 TRAINING data for early stopping
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(len(y))
        n_in = max(1, int(self.hp["inner_val_frac"] * len(y)))
        iv, tr = perm[:n_in], perm[n_in:]

        Xtr = torch.from_numpy(X[tr]).to(dev); ytr = torch.from_numpy(y[tr]).to(dev)
        Xiv = torch.from_numpy(X[iv]).to(dev); yiv = torch.from_numpy(y[iv]).to(dev)

        self.net = SmallCNN(self.hp["channels"], self.hp["hidden"]).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hp["lr"],
                               weight_decay=self.weight_decay)
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        bs = self.hp["batch_size"]

        best_acc, best_state, bad = -1.0, None, 0
        self.history_ = []
        for epoch in range(self.hp["max_epochs"]):
            self.net.train()
            order = torch.randperm(len(ytr), generator=g).to(dev)
            tot = 0.0
            for i in range(0, len(order), bs):
                b = order[i:i + bs]
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.net(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            acc = float((self._logits(Xiv).argmax(1) == yiv).float().mean())
            self.history_.append((tot / len(ytr), acc))
            if acc > best_acc + 1e-4:
                best_acc, bad, self.best_epoch_ = acc, 0, epoch + 1
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= self.hp["patience"]:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    @torch.no_grad()
    def _logits(self, Ximg, bs=2048):
        self.net.eval()
        out = [self.net(Ximg[i:i + bs]) for i in range(0, len(Ximg), bs)]
        return torch.cat(out)

    @torch.no_grad()
    def predict_proba(self, X):
        dev = next(self.net.parameters()).device
        Xt = torch.from_numpy(self._to_img(X)).to(dev)
        return F.softmax(self._logits(Xt), dim=1).cpu().numpy().astype(np.float64)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    # ---------- pickling: always store CPU weights ----------
    def __getstate__(self):
        d = self.__dict__.copy()
        if self.net is not None:
            d["net"] = None
            d["_state_cpu"] = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        return d

    def __setstate__(self, d):
        state = d.pop("_state_cpu", None)
        self.__dict__.update(d)
        if state is not None:
            self.net = SmallCNN(self.hp["channels"], self.hp["hidden"])
            self.net.load_state_dict(state)
            self.net.to(_device()).eval()

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


def register(model_registry, reg_levels=None):
    """Add 'cnn' to the shared core's MODEL_REGISTRY (make_model checks it first)."""
    if reg_levels is not None:
        reg_levels["cnn"] = dict(CNN_REG_LEVELS)
    model_registry["cnn"] = lambda reg="default", seed=0: CNNClassifier(
        weight_decay=CNN_REG_LEVELS[reg], seed=seed)
    return model_registry


# =====================================================================
# Optional CNN-only diagnostics (exploratory; NOT pooled with other models)
# =====================================================================
def integrated_gradients_regions(model, X, y, patch, steps=32, baseline=None):
    """Integrated Gradients (black-image baseline unless given), aggregated into regions:
    red channel, green channel, each blue marker patch, rest of blue channel.
    Returns the mean share of |attribution| per region.

    Caveat for the write-up: colour and shape live in the SAME pixels (the stroke is drawn in
    the red OR green channel), so pixel attributions cannot separate 'uses colour' from
    'uses shape'. That is why the attribute-level counterfactual Shapley in the shared core is
    the primary SHAP-style score; this is only a sanity view."""
    net = model.net
    dev = next(net.parameters()).device
    x = torch.from_numpy(model._to_img(X)).to(dev)
    b = torch.zeros_like(x) if baseline is None else torch.from_numpy(model._to_img(baseline)).to(dev)
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(dev)
    total = torch.zeros_like(x)
    for a in torch.linspace(0, 1, steps, device=dev):
        xi = (b + a * (x - b)).requires_grad_(True)
        out = net(xi).gather(1, yt[:, None]).sum()
        total += torch.autograd.grad(out, xi)[0]
    attr = ((x - b) * total / steps).abs().detach().cpu().numpy()
    mass = attr.reshape(len(attr), -1).sum(1) + 1e-12
    blue = attr[:, 2].copy()
    shares = {"red": attr[:, 0].reshape(len(attr), -1).sum(1) / mass,
              "green": attr[:, 1].reshape(len(attr), -1).sum(1) / mass}
    for name, (r, c) in patch.items():
        shares[name] = blue[:, r, c].reshape(len(blue), -1).sum(1) / mass
        blue[:, r, c] = 0
    shares["blue_other"] = blue.reshape(len(blue), -1).sum(1) / mass
    return {k: float(v.mean()) for k, v in shares.items()}


def realism_auroc(X_cf, X_nat, seed=0, epochs=5):
    """Realism check from the plan: can a CNN tell counterfactual images from NATURAL images
    with the same label and attribute value? Train on half, report AUROC on the other half.
    AUROC ~ 0.5 => counterfactuals are indistinguishable from real data."""
    from sklearn.metrics import roc_auc_score
    n = min(len(X_cf), len(X_nat))
    rng = np.random.default_rng(seed)
    X = np.concatenate([X_cf[rng.permutation(len(X_cf))[:n]],
                        X_nat[rng.permutation(len(X_nat))[:n]]])
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    p = rng.permutation(len(y)); X, y = X[p], y[p]
    h = len(y) // 2
    clf = CNNClassifier(seed=seed, max_epochs=epochs, patience=epochs)
    clf.fit(X[:h].reshape(h, -1), y[:h])
    s = clf.predict_proba(X[h:].reshape(len(y) - h, -1))[:, 1]
    return float(roc_auc_score(y[h:], s)), n
