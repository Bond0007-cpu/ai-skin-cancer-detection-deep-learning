"""
AI Skin Cancer Detection — Training Script
Supports: EfficientNetB4, ResNet50, DenseNet121
Two-phase training: feature extraction → fine-tuning
"""

import argparse
import os
import numpy as np
import mlflow
import mlflow.keras
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    CSVLogger, TensorBoard
)

from src.models.efficientnet import build_efficientnet
from src.models.resnet import build_resnet
from src.models.densenet import build_densenet
from src.data.dataset import build_dataset

CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']


def get_model(name: str, num_classes: int = 7):
    builders = {
        'efficientnetb4': build_efficientnet,
        'resnet50':        build_resnet,
        'densenet121':     build_densenet,
    }
    if name not in builders:
        raise ValueError(f"Unknown model: {name}. Choose from {list(builders.keys())}")
    return builders[name](num_classes=num_classes, input_shape=(224, 224, 3))


def build_callbacks(checkpoint_dir: str, log_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='val_auc', mode='max', save_best_only=True, verbose=1
        ),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(os.path.join(log_dir, 'training.csv')),
        TensorBoard(log_dir=os.path.join(log_dir, 'tensorboard'), histogram_freq=1),
    ]


def train(args):
    # ── Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds, y_train_labels = build_dataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=(224, 224),
        augment=True,
    )

    # ── Class weights ─────────────────────────────────────────────────────
    weights = compute_class_weight('balanced',
                                   classes=np.arange(len(CLASS_NAMES)),
                                   y=y_train_labels)
    class_weight_dict = dict(enumerate(weights))
    print("Class weights:", {CLASS_NAMES[k]: round(v, 3) for k, v in class_weight_dict.items()})

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(args.model)
    callbacks = build_callbacks('checkpoints', 'logs')

    with mlflow.start_run(experiment_id=args.experiment):
        mlflow.log_params({
            "model": args.model,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        })

        # Phase 1 — Feature extraction (frozen base)
        print("\n📌 Phase 1: Feature extraction (frozen base, 20 epochs)")
        model.layers[0].trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        model.fit(train_ds, validation_data=val_ds,
                  epochs=20, callbacks=callbacks,
                  class_weight=class_weight_dict)

        # Phase 2 — Fine-tuning (unfreeze last 20 layers)
        print("\n🔓 Phase 2: Fine-tuning (last 20 layers, full epochs)")
        model.layers[0].trainable = True
        for layer in model.layers[0].layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=args.lr),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        history = model.fit(train_ds, validation_data=val_ds,
                            epochs=args.epochs, callbacks=callbacks,
                            class_weight=class_weight_dict)

        # ── Log final metrics ──────────────────────────────────────────────
        best_val_auc = max(history.history.get('val_auc', [0]))
        best_val_acc = max(history.history.get('val_accuracy', [0]))
        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.keras.log_model(model, "model")

        print(f"\n✅ Training complete!")
        print(f"   Best val AUC:      {best_val_auc:.4f}")
        print(f"   Best val Accuracy: {best_val_acc:.4f}")
        print(f"   Checkpoint saved to: checkpoints/best_model.h5")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train skin cancer detection model')
    parser.add_argument('--model',      default='efficientnetb4',
                        choices=['efficientnetb4', 'resnet50', 'densenet121'])
    parser.add_argument('--data_dir',   default='data/processed')
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--experiment', default='skin_cancer_detection')
    parser.add_argument('--resume',     default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    train(args)
