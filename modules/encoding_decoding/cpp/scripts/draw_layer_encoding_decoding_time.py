import matplotlib.pyplot as plt
import numpy as np

paths_dict = {
	'full-precision': 'fp.results',
	'bits-sharing $\epsilon_{32}$': 'test1.results',
	'bits-sharing $\epsilon_{16}$': 'test2.results',
	'bits-sharing $\epsilon_8$': 'test3.results',
	'bits-sharing $\epsilon_4$': 'test4.results',
	'bits-sharing $\epsilon_4^*$': 'test6.results',
	'bits-sharing $\epsilon_4 & \epsilon_8$': 'test5.results',
}


dark_colors = ['#1F77B4', '#FF7F0F', '#2BA02B', '#D62727', '#9467BD', '#8C564C', '#E377C3', '#7F7F7F']
light_colors = ['#AEC7E8', '#FFBB78', '#99DF8A', '#FF9896', '#C4B0D5', '#C49C94', '#F7B6D2', '#C6C6C6']

# $53.3%\,,\,6.66%$': 'test1.results',
# $26.6%\,,\,6.52%$': 'test2.results',
# $13.3%\,,\,6.66%$': 'test3.results',
# $6.66%\,,\,6.66%$': 'test4.results',
# $12.0%\,,\,21.6%$': 'test6.results',
# $7.00%\,,\,12.0%$': 'test5.results',

def draw_bar(time_list, time_list_8bits=None):
  def format(time_list):
    res = {}
    for item in time_list:
      if 'features.0.components' in item[0]:
        if '0-conv-64' in res:
          for i in range(6):
            res['0-conv-64'][i] += item[i + 2]
        else:
          res['0-conv-64'] = [item[2], item[3], item[4], item[5], item[6], item[7]]
      elif 'features.4.components' in item[0]:
        if '1-conv-128' in res:
          for i in range(6):
            res['1-conv-128'][i] += item[i + 2]
        else:
          res['1-conv-128'] = [item[2], item[3], item[4], item[5], item[6], item[7]]
      elif 'features.8.components' in item[0]:
        if '2-conv-256-1' in res:
          for i in range(6):
            res['2-conv-256-1'][i] += item[i + 2]
        else:
          res['2-conv-256-1'] = [item[2], item[3], item[4], item[5], item[6], item[7]]
      elif 'features.11.components' in item[0]:
        if '3-conv-256-2' in res:
          for i in range(6):
            res['3-conv-256-2'][i] += item[i + 2]
        else:
          res['3-conv-256-2'] = [item[2], item[3], item[4], item[5], item[6], item[7]]
      elif 'features.15.components' in item[0]:
        if '4-conv-512-1' in res:
          for i in range(6):
            res['4-conv-512-1'][i] += item[i + 2]
        else:
          res['4-conv-512-1'] = [item[2], item[3], item[4], item[5], item[6], item[7]]
      elif 'features.18.components' in item[0]:
        if '5-conv-512-2' in res:
          for i in range(6):
            res['5-conv-512-2'][i] += item[i + 2]
        else:
          res['5-conv-512-2'] = [item[2], item[3], item[4], item[5], item[6], item[7]]
      elif 'features.22.components' in item[0]:
        if '6-conv-512-3' in res:
          for i in range(6):
            res['6-conv-512-3'][i] += item[i + 2]
        else:
          res['6-conv-512-3'] = [item[2], item[3], item[4], item[5], item[6], item[7]]
      elif 'features.25.components' in item[0]:
        if '7-conv-512-4' in res:
          for i in range(6):
            res['7-conv-512-4'][i] += item[i + 2]
        else:
          res['7-conv-512-4'] = [item[2], item[3], item[4], item[5], item[6], item[7]]
      elif 'dense1.components' in item[0]:
        if '8-dense-256' in res:
          for i in range(6):
            res['8-dense-256'][i] += item[i + 2]
        else:
          res['8-dense-256'] = [item[2], item[3], item[4], item[5], item[6], item[7]]
      elif 'dense2.components' in item[0]:
        if '9-dense-10' in res:
          for i in range(6):
            res['9-dense-10'][i] += item[i + 2]
        else:
          res['9-dense-10'] = [item[2], item[3], item[4], item[5], item[6], item[7]]
    data = []
    for k, v in res.items():
      data.append([k] + v)
    data.sort(key=lambda x: x[0])
    return data

  time_list = format(time_list)
  if time_list_8bits != None:
    time_list_8bits = format(time_list_8bits)
  layer_name = []
  encode_time = []
  decode_time = []
  metadata = []
  pattern = []
  quantization = []
  layer_size = []
  layer_size_8bits = []
  layer_size_full = [221184, 9437184, 37748736, 75497472, 150994944, 301989888, 301989888, 301989888, 16777216, 327680]
  impr_full = []
  impr_8bits = []
  x = []
  for i, item in enumerate(time_list):
    x.append(i + 1)
    layer_name.append(item[0][2:])
    encode_time.append(item[1])
    decode_time.append(item[2])
    layer_size.append((item[4] + item[5] + item[6]))
    metadata.append(item[4])
    pattern.append(item[5])
    quantization.append(item[6])
    if time_list_8bits != None:
      layer_size_8bits.append(time_list_8bits[i][6])
    layer_size_full[i] /= (8 * 1024 * 1024)
    if time_list_8bits != None:
      impr_8bits.append(layer_size[-1] / layer_size_8bits[-1])
    impr_full.append(layer_size[-1] / layer_size_full[-1])
  x = np.array(x)
  metadata = np.array(metadata)
  pattern = np.array(pattern)
  quantization = np.array(quantization)
  layer_size_full = np.array(layer_size_full)
  layer_size_8bits = np.array(layer_size_8bits)
  print('Impr Full\t{}%'.format(100. * (np.sum(layer_size_full) - np.sum(layer_size)) / np.sum(layer_size_full)))
  print('Impr 8-bit\t{}%'.format(100. * (np.sum(layer_size_8bits) - np.sum(layer_size)) / np.sum(layer_size_8bits)))
  print('Avg Full\t{}'.format(np.average(impr_full[:-1])))
  print('Avg 8-bit\t{}'.format(np.average(impr_8bits[:-1])))

  width = 0.3
  fig, (ax1) = plt.subplots(1, 1, figsize=(6,3), constrained_layout=True)
  ax1.bar(layer_name, decode_time, width=width, color=light_colors[6])
  plt.xticks(rotation=30)
  ax1.set_xlabel('Layer')
  ax1.set_ylabel('Time (s)')
  plt.savefig('decoding.pdf')
  plt.close()

  fig, (ax1) = plt.subplots(1, 1, figsize=(6,3), constrained_layout=True)
  # ax1.bar(x - width / 2, encode_time, width=width, color=light_colors[6])
  # ax1.bar(x + width / 2, decode_time, width=width, color=light_colors[5])
  ax1.bar(layer_name, encode_time, width=width, color=light_colors[6])
  plt.xticks(rotation=30)
  ax1.set_xlabel('Layer')
  ax1.set_ylabel('Time (s)')
  plt.savefig('encoding.pdf')
  plt.close()

  def add_labels(rects, text):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height, text[i], ha='center', va='bottom')
 
  print('Metadata', metadata)
  print('Pattern', pattern)
  print('Quantization', quantization)
  print('BitEnsemble', layer_size)
  print('Full-P', layer_size_full)
  if time_list_8bits != None:
    print('8-Bit', layer_size_8bits)
  labels = ('BitsEnsemble (M)', 'BitsEnsemble (P)', 'BitsEnsemble (Q)', 'Full-rank (full-precision)', 'Full-rank (8 bit)')
  fig, (ax1) = plt.subplots(1, 1, figsize=(6,3), constrained_layout=True)
  ax1.bar(x - width, metadata, width=width, color=light_colors[3], label=labels[0])
  ax1.bar(x - width, pattern, bottom=metadata, width=width, color=light_colors[5], label=labels[0])
  p = ax1.bar(x - width, quantization, bottom=metadata + pattern, width=width, color=light_colors[6], label=labels[0])
  prop = ((metadata + pattern + quantization) * 100 / layer_size_full).astype(int).astype(str)
  add_labels(p, prop)
  ax1.bar(x, layer_size_full, width=width, color=light_colors[0], label=labels[1])
  if time_list_8bits != None:
    p = ax1.bar(x + width, layer_size_8bits, width=width, color=light_colors[2], label=labels[2])
    prop = (layer_size_8bits * 100 / layer_size_full).astype(int).astype(str)
    add_labels(p, prop)
  lgd = fig.legend(labels, loc='lower right', bbox_to_anchor=(1,-0.15), ncol=3, bbox_transform=fig.transFigure, frameon=False)
  plt.ylim((0, 20))
  ax1.set_xticks(x)
  ax1.set_xticklabels(layer_name, rotation=30)
  # plt.xticks(rotation=30)
  # ax1.set_xticks(x, layer_name)
  ax1.set_xlabel('Layer')
  ax1.set_ylabel('Size (MB)')
  plt.savefig('size.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
  plt.close()




