from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")

text = [
    "Angkatan Bersenjata Kanada. 1 Misi penjaga perdamaian Kanada berskala besar pertama dimulai di Mesir pada 24 November 1956. 2 Ada sekitar 65.000 Pasukan Reguler dan 25.000 anggota cadangan di militer Kanada. 3 Di Kanada, 9 Agustus ditetapkan sebagai Hari Penjaga Perdamaian Nasional.",
    "Ichthyodes rufipes adalah spesies kumbang tanduk panjang yang berasal dari famili Cerambycidae. Spesies ini juga merupakan bagian dari genus Ichthyodes, ordo Coleoptera, kelas Insecta, filum Arthropoda, dan kingdom Animalia.", 
    "suhu rata-rata di london dalam bulan Agustus dalam selsius"
]

a = tokenizer(text)

for text_1 in text:
    b = tokenizer.tokenize(text_1)
    print(f"==>> b: {b}")
    print(len(b))

print(f"==>> a: {a}")


