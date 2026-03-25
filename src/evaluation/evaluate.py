"""
Model evaluation script — generates full clinical-grade evaluation report.
Includes: classification report, confusion matrix, AUC-ROC, Cohen's Kappa, Grad-CAM samples.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, cohen_kappa_score, roc_curve, auc
)

from src.data.dataset import build_test_dataset
from src.models.gradcam import predict_with_explanation

CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']


def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and test data
    model = tf.keras.models.load_model(args.checkpoint)
    test_ds, y_true, images = build_test_dataset(args.test_dir, return_images=True)

    print(f"Evaluating on {len(y_true)} test samples...")

    # Predictions
    y_pred_proba = model.predict(test_ds, verbose=1)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    y_true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true

    # ── Metrics ──────────────────────────────────────────────────────────
    report = classification_report(y_true_labels, y_pred,
                                   target_names=CLASS_NAMES, output_dict=True)
    kappa  = cohen_kappa_score(y_true_labels, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')

    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(classification_report(y_true_labels, y_pred, target_names=CLASS_NAMES))
    print(f"AUC-ROC (macro): {auc_roc:.4f}")
    print(f"Cohen's Kappa:   {kappa:.4f}")

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_true_labels, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix — Skin Cancer Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved: {cm_path}")

    # ── ROC curves ───────────────────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{cls} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Class')
    plt.legend(loc='lower right')
    plt.tight_layout()
    roc_path = os.path.join(args.output_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"✅ ROC curves saved: {roc_path}")

    # ── Grad-CAM samples ─────────────────────────────────────────────────
    if args.grad_cam:
        gradcam_dir = os.path.join(args.output_dir, 'gradcam_samples')
        os.makedirs(gradcam_dir, exist_ok=True)
        print("Generating Grad-CAM samples...")
        for i in range(min(10, len(images))):
            pred_cls, conf, overlay, _ = predict_with_explanation(
                model, images[i], model_name='efficientnetb4'
            )
            true_cls = CLASS_NAMES[y_true_labels[i]]
            plt.imsave(
                os.path.join(gradcam_dir, f'sample_{i:03d}_{true_cls}_pred_{pred_cls}.png'),
                overlay
            )
        print(f"✅ Grad-CAM samples saved: {gradcam_dir}")

    # ── Save JSON report ─────────────────────────────────────────────────
    summary = {
        'accuracy':      report['accuracy'],
        'auc_roc_macro': auc_roc,
        'cohens_kappa':  kappa,
        'per_class':     {cls: report[cls] for cls in CLASS_NAMES},
    }
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Full report saved: {report_path}")

    # Warn if targets not met
    if auc_roc < 0.97:
        print(f"⚠️  AUC-ROC {auc_roc:.4f} < target 0.97 — consider more training or data")
    if report['accuracy'] < 0.95:
        print(f"⚠️  Accuracy {report['accuracy']:.4f} < target 0.95")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_dir',   default='data/processed/test')
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--grad_cam',   type=bool, default=True)
    evaluate(parser.parse_args())
