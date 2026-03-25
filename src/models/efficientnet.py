"""
EfficientNetB4-based skin lesion classifier.
Two-phase training: frozen base → fine-tune last 20 layers.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB4

NUM_CLASSES = 7
CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']


def build_efficientnet(num_classes: int = NUM_CLASSES,
                       input_shape: tuple = (224, 224, 3),
                       dropout_rate: float = 0.5) -> Model:
    """
    Build EfficientNetB4 with custom classification head.

    Architecture:
        EfficientNetB4 (ImageNet pretrained, frozen)
        → GlobalAveragePooling2D
        → Dense(512, ReLU)
        → BatchNormalization
        → Dropout(0.5)
        → Dense(num_classes, Softmax)
    """
    inputs = layers.Input(shape=input_shape, name='image_input')

    # Base model — frozen initially, unfreeze last 20 layers in Phase 2
    base = EfficientNetB4(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
    )
    base.trainable = False  # Phase 1: feature extraction

    # Custom classification head
    x = base.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(512, activation='relu', name='dense_512')(x)
    x = layers.BatchNormalization(name='bn')(x)
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs, name='EfficientNetB4_SkinCancer')

    # Compile with AUC metric (key metric for imbalanced medical data)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )

    return model


def unfreeze_top_layers(model: Model, n_layers: int = 20) -> Model:
    """Unfreeze last n layers of EfficientNetB4 for fine-tuning (Phase 2)."""
    base = model.get_layer('efficientnetb4')
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    print(f"✅ Unfroze last {n_layers} layers for fine-tuning")
    return model
