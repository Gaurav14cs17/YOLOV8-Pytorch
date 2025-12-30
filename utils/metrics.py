"""Metrics computation for object detection evaluation."""

import numpy as np


def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
    """Apply box filter smoothing.

    Args:
        y: Input array
        f: Smoothing fraction

    Returns:
        Smoothed array
    """
    nf = round(len(y) * f * 2) // 2 + 1  # filter elements (must be odd)
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def compute_ap(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray,
               target_cls: np.ndarray, eps: float = 1e-16) -> tuple:
    """Compute Average Precision.

    Args:
        tp: True positives (N, num_iou_thresholds)
        conf: Confidence scores (N,)
        pred_cls: Predicted classes (N,)
        target_cls: Ground truth classes

    Returns:
        Tuple of (tp, fp, precision, recall, mAP@0.5, mAP@0.5:0.95)
    """
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    # Create Precision-Recall curve and compute AP
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    ap = np.zeros((nc, tp.shape[1]))
    px = np.linspace(0, 1, 1000)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs

        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))
            x = np.linspace(0, 1, 101)
            ap[ci, j] = np.trapz(np.interp(x, m_rec, m_pre), x)

    # Compute F1
    f1 = 2 * p * r / (p + r + eps)
    i = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]

    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    ap50, ap = ap[:, 0], ap.mean(1)
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), ap.mean()

    return tp, fp, m_pre, m_rec, map50, mean_ap


class MetricsCalculator:
    """Class for computing detection metrics."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.all_tp = []
        self.all_conf = []
        self.all_pred_cls = []
        self.all_target_cls = []

    def update(self, tp: np.ndarray, conf: np.ndarray,
               pred_cls: np.ndarray, target_cls: np.ndarray):
        """Update with batch results."""
        self.all_tp.append(tp)
        self.all_conf.append(conf)
        self.all_pred_cls.append(pred_cls)
        self.all_target_cls.append(target_cls)

    def compute(self) -> dict:
        """Compute final metrics."""
        if not self.all_tp:
            return {'mAP50': 0.0, 'mAP': 0.0, 'precision': 0.0, 'recall': 0.0}

        tp = np.concatenate(self.all_tp)
        conf = np.concatenate(self.all_conf)
        pred_cls = np.concatenate(self.all_pred_cls)
        target_cls = np.concatenate(self.all_target_cls)

        _, _, precision, recall, map50, mean_ap = compute_ap(
            tp, conf, pred_cls, target_cls
        )

        return {
            'mAP50': map50,
            'mAP': mean_ap,
            'precision': precision,
            'recall': recall
        }
