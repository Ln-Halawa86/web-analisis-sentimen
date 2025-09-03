import os

def remove_duplicate_lines(lines):
    """Menghapus baris duplikat dari list string"""
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    return unique_lines

def validate_line(line):
    """Memvalidasi bahwa baris hanya punya 2 kolom"""
    parts = line.strip().split('\t')
    return len(parts) == 2

folder = 'kamus'

for filename in os.listdir(folder):
    if filename.endswith('.txt') and not filename.endswith('_tab.txt'):
        input_path = os.path.join(folder, filename)
        output_path = os.path.join(folder, filename.replace('.txt', '_tab.txt'))

        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.readlines()

        # Hapus baris kosong & whitespace
        cleaned_content = [line.strip() for line in content if line.strip()]

        # Hapus baris duplikat
        unique_content = remove_duplicate_lines(cleaned_content)

        # Validasi jumlah kolom dan hapus yang tidak sesuai
        valid_content = []
        invalid_lines = []

        for line in unique_content:
            if validate_line(line):
                valid_content.append(line)
            else:
                invalid_lines.append(line)

        # Jika ada baris tidak valid, tampilkan pesan
        if invalid_lines:
            print(f"\n[⚠️ Baris tidak valid di {filename}]")
            for line in invalid_lines:
                print(f"→ {line}")

        # Ganti koma jadi tab
        modified_content = [line.replace(',', '\t') for line in valid_content]

        # Simpan ke file baru
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(modified_content) + '\n')

        print(f"\n✅ File diproses dan duplikasi dihapus: {output_path}")