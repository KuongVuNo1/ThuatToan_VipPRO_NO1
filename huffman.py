import heapq
import os
from collections import Counter

class HuffmanNode:
  def __init__(self, char, freq):
    self.char = char
    self.freq = freq
    self.left = None
    self.right = None

  def __lt__(self, other):
    return self.freq < other.freq

class HuffmanCoding:
  def __init__(self):
    self.codes = {}
    self.reverse_codes = {}

  def build_tree(self, text):
    frequency = Counter(text)
    heap = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
      root = HuffmanNode(None, heap[0].freq)
      root.left = heapq.heappop(heap)
      return root

    while len(heap) > 1:
      node1 = heapq.heappop(heap)
      node2 = heapq.heappop(heap)
      merged = HuffmanNode(None, node1.freq + node2.freq)
      merged.left = node1
      merged.right = node2
      heapq.heappush(heap, merged)

    return heap[0]

  def generate_codes(self, root, current_code=""):
    if root is None:
      return
    if root.char is not None:
      self.codes[root.char] = current_code
      self.reverse_codes[current_code] = root.char
      return
    self.generate_codes(root.left, current_code + "0")
    self.generate_codes(root.right, current_code + "1")

  def compress(self, input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
      text = f.read()
    if not text:
      open(output_file, "wb").close()
      return

    root = self.build_tree(text)
    self.generate_codes(root)

    encoded_text = "".join(self.codes[char] for char in text)
    padding = (8 - len(encoded_text) % 8) % 8
    encoded_text += "0" * padding

    header_info = f"{padding}:{len(self.codes)}
"
    for char, code in self.codes.items():
      char_repr = repr(char)
      header_info += f"{char_repr} {code}
"

    b_arr = bytearray()
    for i in range(0, len(encoded_text), 8):
      b_arr.append(int(encoded_text[i : i + 8], 2))

    with open(output_file, "wb") as f:
      f.write(header_info.encode("utf-8"))
      f.write(b"---DATA---
")
      f.write(bytes(b_arr))

  def decompress(self, input_file, output_file):
    with open(input_file, "rb") as f:
      content = f.read()

    if not content:
      open(output_file, "w", encoding="utf-8").close()
      return

    header_part, data_part = content.split(b"---DATA---
", 1)
    header_lines = header_part.decode("utf-8").split("
")

    padding, code_count = map(int, header_lines[0].split(":"))
    reverse_codes = {}
    for i in range(1, code_count + 1):
      line = header_lines[i]
      char_repr, code = line.rsplit(" ", 1)
      reverse_codes[code] = eval(char_repr)

    bit_string = "".join(f"{byte:08b}" for byte in data_part)
    if padding > 0:
      bit_string = bit_string[:-padding]

    current_code = ""
    decoded_text = []
    for bit in bit_string:
      current_code += bit
      if current_code in reverse_codes:
        decoded_text.append(reverse_codes[current_code])
        current_code = ""

    with open(output_file, "w", encoding="utf-8") as f:
      f.write("".join(decoded_text))

if __name__ == "__main__":
  sample_text = "Thuat toan nang cao - Data Compression with Huffman Codes."
  with open("sample.txt", "w", encoding="utf-8") as f:
    f.write(sample_text)
  huffman = HuffmanCoding()
  huffman.compress("sample.txt", "compressed.bin")
  huffman.decompress("compressed.bin", "output.txt")
  print(f"Kich thuoc goc: {os.path.getsize('sample.txt')} bytes")
  print(f"Kich thuoc sau nen: {os.path.getsize('compressed.bin')} bytes")
