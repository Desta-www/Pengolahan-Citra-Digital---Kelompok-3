import os
import cv2
import numpy as np
from skimage import restoration, img_as_float

# ============================================================
# KONFIGURASI
# ============================================================

INPUT_ROOT  = "RestorasiCitra_TugasKelompok3" # Folder utama berisi subfolder-subfolder
OUTPUT_ROOT = "Hasil_Wiener_All_Sigma"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

PSF_SIZE = 1      # Harus integer, ukuran kernel PSF (piksel)
BALANCE  = 0.1


# ============================================================
# FUNGSI MEMBUAT PSF
# ============================================================

def create_average_psf(size=1):
    """
    Membuat PSF average blur.
    size harus integer ganjil (3, 5, 7, ...).
    """
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"PSF size harus integer >= 1, dapat: {size}")
    psf = np.ones((size, size), dtype=np.float32)
    psf /= np.sum(psf)
    return psf


# ============================================================
# FUNGSI WIENER FILTER
# ============================================================

def wiener_filter_grayscale(image_gray, psf, balance=0.1):
    image_float = img_as_float(image_gray)
    restored    = restoration.wiener(image_float, psf, balance=balance, clip=False)
    restored    = np.clip(restored, 0, 1)
    return (restored * 255).astype(np.uint8)


def wiener_filter_color(image_bgr, psf, balance=0.1):
    channels = cv2.split(image_bgr)
    restored_channels = [
        wiener_filter_grayscale(ch, psf, balance) for ch in channels
    ]
    return cv2.merge(restored_channels)


# ============================================================
# FUNGSI PROSES — REKURSIF MENDALAM
# ============================================================

def process_all_images(input_root, output_root, psf, balance=0.1):
    """
    Struktur yang didukung (os.walk sudah rekursif otomatis):

        input_root/
        ├── kategori_A/
        │   ├── gambar1.jpg
        │   └── gambar2.png
        ├── kategori_B/
        │   ├── sub_sub/
        │   │   └── gambar3.jpg
        │   └── gambar4.jpeg
        └── gambar_langsung.jpg   ← file di root juga ikut diproses

    Struktur folder direplikasi persis di output_root.
    """
    os.makedirs(output_root, exist_ok=True)

    dataset_rows = []
    total_success = 0
    total_failed  = 0
    failed_files  = []

    # os.walk sudah rekursif — masuk ke semua subfolder secara otomatis
    for root, dirs, files in os.walk(input_root):
        image_files = [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]

        if not image_files:
            continue  # Lewati folder kosong / tidak ada gambar

        # Hitung kedalaman relatif untuk log yang informatif
        relative_folder = os.path.relpath(root, input_root)
        depth_label     = relative_folder if relative_folder != "." else "(root)"
        print(f"\n  Folder: {depth_label} ({len(image_files)} gambar)")

        # Buat folder output yang sesuai
        output_folder = os.path.join(output_root, relative_folder)
        os.makedirs(output_folder, exist_ok=True)

        for filename in image_files:
            input_path  = os.path.join(root, filename)
            name, ext   = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}_wiener.jpeg")

            image = cv2.imread(input_path)
            if image is None:
                print(f"    [SKIP] Tidak bisa dibaca: {filename}")
                total_failed += 1
                failed_files.append(input_path)
                continue

            try:
                restored = wiener_filter_color(image, psf, balance)
                if cv2.imwrite(output_path, restored):
                    print(f"    [OK]   {filename} -> {output_path}")
                    total_success += 1
                    dataset_rows.append({
                        "original_path" : input_path,
                        "wiener_path"   : output_path,
                        "folder"        : relative_folder,
                        "filename"      : filename,
                        "psf_size"      : PSF_SIZE,
                        "balance"       : balance,
                    })
                else:
                    raise RuntimeError("cv2.imwrite gagal menulis file")

            except Exception as e:
                print(f"    [ERR]  {filename} | {e}")
                total_failed += 1
                failed_files.append(f"{input_path} | Error: {e}")

    return dataset_rows, total_success, total_failed, failed_files


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    print("=" * 50)
    print("RESTORASI GAMBAR — WIENER FILTER")
    print("=" * 50)
    print(f"Input  : {INPUT_ROOT}")
    print(f"Output : {OUTPUT_ROOT}")
    print(f"PSF    : {PSF_SIZE}x{PSF_SIZE} average blur")
    print(f"Balance: {BALANCE}")

    if not os.path.exists(INPUT_ROOT):
        print(f"\n[ERROR] Folder '{INPUT_ROOT}' tidak ditemukan.")
        return

    psf = create_average_psf(size=PSF_SIZE)

    rows, success, failed, failed_files = process_all_images(
        input_root  = INPUT_ROOT,
        output_root = OUTPUT_ROOT,
        psf         = psf,
        balance     = BALANCE,
    )

    print()
    print("=" * 50)
    print("SELESAI")
    print("=" * 50)
    print(f"Berhasil : {success} gambar")
    print(f"Gagal    : {failed} gambar")

    if failed_files:
        print("\nDaftar gagal:")
        for item in failed_files:
            print(f"  - {item}")

    print(f"\nOutput tersimpan di: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()