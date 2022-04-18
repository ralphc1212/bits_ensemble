import os
import sys
import math
from draw_layer_encoding_decoding_time import *

def encode_size(encoded_dir, layer_name, buffer_size, num_data):
  file = os.path.join(encoded_dir, 'encoded_{}_{}_{}'.format(buffer_size, num_data, layer_name.replace('.in', '.bin')))
  return os.stat(file).st_size

def collect_results(file_path, encoded_dir, pretty=True):
  f = open(file_path, 'r')
  lines = f.readlines()
  f.close()
  res_map = {}
  for line in lines:
    line = line.strip()
    if line == "":
      continue
    items = line.split()
    layer_name = items[0]
    buffer_size = int(items[1])
    num_data = int(items[2])
    load_time = float(items[3])
    encode_time = float(items[4])
    decode_time = float(items[5])
    metadata = float(items[6]) / (1024 * 1024)
    pattern = float(items[7]) / (1024 * 1024)
    quantization = float(items[8]) / (1024 * 1024)
    layer_size = encode_size(encoded_dir, layer_name, buffer_size, num_data) / (1024 * 1024)
    if (buffer_size, num_data) not in res_map:
      res_map[(buffer_size, num_data)] = [(layer_name, load_time, encode_time, decode_time, layer_size, metadata, pattern, quantization)]
    else:
      res_map[(buffer_size, num_data)].append((layer_name, load_time, encode_time, decode_time, layer_size, metadata, pattern, quantization))
  for hp, res in res_map.items():
    if pretty:
      print('Buffer Size\t{}'.format(hp[0]))
      print('# Block Data\t{}'.format(hp[1]))
    else:
      print(hp[0], '\t', hp[1])
    sum_load_time = 0
    sum_encode_time = 0
    sum_decode_time = 0
    sum_encoded_size = 0
    sum_metadata_size = 0
    sum_pattern_size = 0
    sum_quantization_size = 0
    for d in res:
      sum_load_time += d[1]
      sum_encode_time += d[2]
      sum_decode_time += d[3]
      sum_encoded_size += (d[5] + d[6] + d[7])
      sum_metadata_size += d[5]
      sum_pattern_size += d[6]
      sum_quantization_size += d[7]
    if pretty:
      print('Load Time\t{} (s)'.format(sum_load_time))
      print('Encode Time\t{} (s)'.format(sum_encode_time))
      print('Decode Time\t{} (s)'.format(sum_decode_time))
      print('Encoded Size\t{} (B)'.format(sum_encoded_size))
      print('Metadata\t{:.2f}%'.format(100 * sum_metadata_size / sum_encoded_size))
      print('Pattern\t{:.2f}%'.format(100 * sum_pattern_size / sum_encoded_size))
      print('Quantization\t{:.2f}%'.format(100 * sum_quantization_size / sum_encoded_size))
    else:
      print(sum_load_time, '\t', sum_encode_time, '\t', sum_decode_time, '\t', sum_encoded_size)
  return res_map

if __name__ == '__main__':
  if len(sys.argv) < 4:
    sys.exit('usage: ./print_result.py (nn name) (encoded files dir) (time results dir)')
  
  nn = sys.argv[1]
  encoded_dir = sys.argv[2]
  time_dir = sys.argv[3]
  print_8bits = True
  time_file_path = os.path.join(time_dir, nn + '.out')
  res_map = collect_results(time_file_path, encoded_dir)
  if print_8bits:
    time_file_path_8bits = os.path.join(time_dir, nn + '-8.out')
    res_map_8bits = collect_results(time_file_path_8bits, encoded_dir)
  draw_bar(res_map[(2048, 256)], res_map_8bits[(2048, 256)] if print_8bits else None)
