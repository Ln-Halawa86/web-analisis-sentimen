# sentiment_labeler.py
import os

class SentimentLabeler:
    def __init__(self, leksikon_path="kamus"):
        self.word_weights = {}  # Akan menyimpan bobot akumulatif (net effect)
        self._load_lexicon(os.path.join(leksikon_path, "kata_positif.txt"))
        self._load_lexicon(os.path.join(leksikon_path, "kata_negatif.txt"))

    def _load_lexicon(self, filepath):
        """Membaca leksikon dengan format 'kata,bobot', dan akumulasikan bobot jika kata sudah ada."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue  # Lewati baris kosong, komentar, atau bullet

                    if ',' not in line:
                        print(f"Peringatan (baris {line_num}): Tidak ada koma di '{line}'")
                        continue

                    parts = line.rsplit(',', 1)  # Pisah dari kanan, sekali saja
                    word = parts[0].strip().lower()
                    try:
                        weight = float(parts[1])
                        
                        # Jika kata sudah ada, tambahkan bobotnya
                        if word in self.word_weights:
                            self.word_weights[word] += weight
                        else:
                            self.word_weights[word] = weight
                            
                    except ValueError:
                        print(f"Peringatan (baris {line_num}): Bobot tidak valid '{parts[1]}' pada '{line}'")
        except FileNotFoundError:
            raise Exception(f"File leksikon tidak ditemukan: {filepath}")

    def label_text(self, text):
        """Memberi label berdasarkan total skor neto dari semua kata yang muncul."""
        text_lower = text.lower()
        total_score = 0.0
        matched_phrases = []

        # Urutkan frasa dari yang terpanjang agar tidak terjadi partial match
        sorted_phrases = sorted(self.word_weights.keys(), key=len, reverse=True)

        for phrase in sorted_phrases:
            if phrase in text_lower:
                count = text_lower.count(phrase)
                net_weight = self.word_weights[phrase]
                total_score += net_weight * count
                matched_phrases.append((phrase, net_weight, count))

        # Tentukan label akhir
        if total_score > 0:
            label = "positif"
        elif total_score < 0:
            label = "negatif"
        else:
            label = "netral"

        return label, total_score, matched_phrases