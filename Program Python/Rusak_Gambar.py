import cv2
import numpy as np
import os
import glob

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = image.astype(np.float64) + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def process_images(input_paths, output_dir='output_noisy', sigma=30):

    # Resolve glob pattern jika input berupa string
    if isinstance(input_paths, str):
        input_paths = glob.glob(input_paths)

    if not input_paths:
        print("Error: Tidak ada gambar yang ditemukan!")
        return

    # Buat folder output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    for path in input_paths:
        image = cv2.imread(path, 1)  # flag 1 = BGR warna penuh

        if image is None:
            print(f"  [SKIP] Tidak bisa membaca: {path}")
            continue

        noisy = add_gaussian_noise(image, sigma=sigma)

        # Buat nama output: tambahkan prefix 'noisy_' pada nama file asli
        filename     = os.path.basename(path)
        name, ext    = os.path.splitext(filename)
        output_name  = f"noisy_{name}{ext}"
        output_path  = os.path.join(output_dir, output_name)

        cv2.imwrite(output_path, noisy)
        print(f"  [OK] {path} -> {output_path}")
        success_count += 1

    print(f"\nSelesai: {success_count}/{len(input_paths)} gambar berhasil diproses.")
    print(f"Output tersimpan di folder: '{output_dir}/'")


# ── Cara penggunaan ──────────────────────────────────────────────────────────

for pattern in ['*.jpeg']:
    for i in range(1, 36):
        process_images(pattern, output_dir=f'output_sigma{i}', sigma=i)