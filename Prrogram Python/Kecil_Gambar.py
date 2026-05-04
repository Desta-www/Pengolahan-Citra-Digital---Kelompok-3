from PIL import Image

# Buka gambar asli
img_asli = Image.open("OPizza.jpeg")

# Tentukan ukuran baru (misal: 1/4 dari ukuran asli)
# Ukuran asli: 1280x960 -> Ukuran baru: 320x240
lebar_baru = 320
tinggi_baru = 240

# Kecilkan gambar
img_kecil = img_asli.resize((lebar_baru, tinggi_baru))

# Simpan sebagai file sementara untuk diproses script kamu
img_kecil.save("Pizza.jpeg")

# Sekarang, gunakan "Foto_Ramen_Kecil.jpeg" sebagai input di script kamu
input_file = "OPizza.jpeg"