import os
from dotenv import load_dotenv

# 1. Muat file .env
load_dotenv() 

# 2. Tarik variabelnya
kunci_saya = os.getenv("GEMINI_API_KEY")

# 3. Cek apakah kodenya berhasil membaca (Jangan bagikan hasil print ini ke publik!)
if kunci_saya:
    # Menggunakan tanda kutip tunggal di sekitar hasil print agar kita bisa melihat jika ada spasi tersembunyi
    print(f"API Key yang terbaca: '{kunci_saya}'")
    print(f"Panjang karakter: {len(kunci_saya)}")
    
    if kunci_saya.startswith('"') or kunci_saya.startswith("'"):
        print("\n⚠️ PERINGATAN: API Key Anda memiliki tanda kutip di awalnya. Tolong hapus tanda kutip di file .env!")
    if kunci_saya.endswith('"') or kunci_saya.endswith("'"):
        print("\n⚠️ PERINGATAN: API Key Anda memiliki tanda kutip di akhirnya. Tolong hapus tanda kutip di file .env!")
    if " " in kunci_saya:
        print("\n⚠️ PERINGATAN: API Key Anda mengandung spasi kosong. Tolong hapus spasi di file .env!")
else:
    print("\n❌ API Key TIDAK TERBACA. Pastikan file bernama persis '.env' (jangan '.env.txt' atau lainnya).")
