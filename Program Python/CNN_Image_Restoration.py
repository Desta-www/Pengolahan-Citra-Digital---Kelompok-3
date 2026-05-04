import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# KONFIGURASI
# ============================================================

NOISY_ROOT  = "Hasil_Wiener_All_Sigma"        # Folder utama berisi subfolder gambar noisy
CLEAN_ROOT  = "Foto_Asli"         # Folder utama berisi subfolder gambar bersih
OUTPUT_ROOT = "hasil_cnn"            # Folder output hasil prediksi
MODEL_PATH  = "model_refinement.h5"  # Path simpan/load model

IMG_WIDTH  = 320    # Lebar gambar (piksel)
IMG_HEIGHT = 240    # Tinggi gambar (piksel)

BATCH_SIZE       = 8
EPOCHS           = 50
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ============================================================
# ARSITEKTUR MODEL
# ============================================================

def build_refinement_cnn(input_shape=(240, 320, 1)):  # (H, W, C)
    inputs = layers.Input(shape=input_shape)

    # Layer 1: Conv + ReLU
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)

    # Layer 2-10: Deep Layers
    for _ in range(10):
        x = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # Prediksi residu noise
    residual = layers.Conv2D(1, (3, 3), padding='same')(x)

    # Output: gambar dikurangi residu
    output = layers.Subtract()([inputs, residual])

    model = models.Model(inputs, output)
    return model


# ============================================================
# FUNGSI LOAD & PREPROCESS GAMBAR
# ============================================================

def load_image(path):
    """
    Baca gambar grayscale, normalisasi ke [0, 1].
    Tidak ada resize karena semua gambar sudah 320x240.
    Jika ada gambar dengan ukuran berbeda, akan di-raise error.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Tidak bisa membaca: {path}")

    h, w = img.shape
    if w != IMG_WIDTH or h != IMG_HEIGHT:
        raise ValueError(
            f"Ukuran tidak sesuai: {path} "
            f"({w}x{h}, expected {IMG_WIDTH}x{IMG_HEIGHT})"
        )

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)   # (H, W, 1)
    return img


# ============================================================
# FUNGSI PENGUMPUL PATH GAMBAR (REKURSIF)
# ============================================================

def collect_image_pairs(noisy_root, clean_root):
    """
    Mengumpulkan pasangan (noisy_path, clean_path) dari dua folder
    dengan struktur subfolder yang identik.

    Struktur yang didukung:
        noisy_root/          clean_root/
        ├── kategori_A/      ├── kategori_A/
        │   ├── img1.jpg     │   ├── img1.jpg
        │   └── img2.png     │   └── img2.png
        └── kategori_B/      └── kategori_B/
            └── img3.jpeg        └── img3.jpeg
    """
    pairs = []

    for root, dirs, files in os.walk(noisy_root):
        image_files = [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]
        if not image_files:
            continue

        relative_folder = os.path.relpath(root, noisy_root)
        clean_folder    = os.path.join(clean_root, relative_folder)

        for filename in image_files:
            noisy_path = os.path.join(root, filename)
            name, _    = os.path.splitext(filename)

            # Cari pasangan clean dengan nama sama (coba semua ekstensi)
            clean_path = None
            for ext in VALID_EXTENSIONS:
                candidate = os.path.join(clean_folder, name + ext)
                if os.path.exists(candidate):
                    clean_path = candidate
                    break

            if clean_path is None:
                print(f"  [SKIP] Tidak ada pasangan clean untuk: {noisy_path}")
                continue

            pairs.append((noisy_path, clean_path))

    print(f"Total pasangan gambar ditemukan: {len(pairs)}")
    return pairs


# ============================================================
# DATASET GENERATOR (tf.data)
# ============================================================

def make_dataset(pairs, batch_size=8, shuffle=True):
    noisy_paths = [p[0] for p in pairs]
    clean_paths = [p[1] for p in pairs]

    def load_pair(noisy_path, clean_path):
        noisy = tf.numpy_function(
            lambda p: load_image(p.decode()),
            [noisy_path], tf.float32
        )
        clean = tf.numpy_function(
            lambda p: load_image(p.decode()),
            [clean_path], tf.float32
        )
        noisy.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
        clean.set_shape([IMG_HEIGHT, IMG_WIDTH, 1])
        return noisy, clean

    ds = tf.data.Dataset.from_tensor_slices((noisy_paths, clean_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs))
    ds = ds.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# FUNGSI SIMPAN HASIL PREDIKSI (REKURSIF)
# ============================================================

def save_predictions(model, noisy_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    count = 0

    for root, dirs, files in os.walk(noisy_root):
        image_files = [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]
        if not image_files:
            continue

        relative_folder = os.path.relpath(root, noisy_root)
        output_folder   = os.path.join(output_root, relative_folder)
        os.makedirs(output_folder, exist_ok=True)

        print(f"\n {relative_folder} ({len(image_files)} gambar)")

        for filename in image_files:
            input_path  = os.path.join(root, filename)
            name, _     = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}_restored.png")

            try:
                img   = load_image(input_path)                   # (H, W, 1)
                batch = np.expand_dims(img, axis=0)              # (1, H, W, 1)
                pred  = model.predict(batch, verbose=0)[0]       # (H, W, 1)
                pred  = np.clip(pred, 0, 1)
                out   = (pred[:, :, 0] * 255).astype(np.uint8)  # (H, W)
                cv2.imwrite(output_path, out)
                print(f"    [OK] {filename}")
                count += 1
            except Exception as e:
                print(f"    [ERR] {filename} | {e}")

    print(f"\nTotal tersimpan: {count} gambar di '{output_root}/'")


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    print("=" * 55)
    print("CNN REFINEMENT — IMAGE DENOISING")
    print("=" * 55)
    print(f"Ukuran gambar : {IMG_WIDTH}x{IMG_HEIGHT} piksel")
    print(f"Input noisy   : {NOISY_ROOT}")
    print(f"Input clean   : {CLEAN_ROOT}")
    print(f"Output        : {OUTPUT_ROOT}")

    # ── 1. Kumpulkan pasangan gambar ────────────────────────
    print("\n[1] Mengumpulkan pasangan gambar...")
    pairs = collect_image_pairs(NOISY_ROOT, CLEAN_ROOT)

    if not pairs:
        print("Tidak ada pasangan gambar. Periksa struktur folder.")
        return

    # ── 2. Split train/val (80:20) ──────────────────────────
    split    = int(len(pairs) * 0.8)
    train_ds = make_dataset(pairs[:split], BATCH_SIZE, shuffle=True)
    val_ds   = make_dataset(pairs[split:], BATCH_SIZE, shuffle=False)
    print(f"\n[2] Train: {split} | Val: {len(pairs) - split}")

    # ── 3. Bangun & compile model ───────────────────────────
    print("\n[3] Membangun model...")
    model = build_refinement_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    model.summary()

    # ── 4. Callbacks ────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, save_best_only=True,
            monitor='val_loss', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10, monitor='val_loss',
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=5,
            monitor='val_loss', verbose=1
        ),
    ]

    # ── 5. Training ─────────────────────────────────────────
    print(f"\n[4] Training ({EPOCHS} epoch maks)...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ── 6. Simpan prediksi ───────────────────────────────────
    print("\n[5] Menyimpan hasil prediksi...")
    save_predictions(model, NOISY_ROOT, OUTPUT_ROOT)

    print("\nSelesai.")


if __name__ == "__main__":
    main()