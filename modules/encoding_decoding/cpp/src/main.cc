#include "common.h"
#include "bit_ensemble_4bits.h"

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cout << "Please input: ./bit_ensemble (buffer size) (num block data) (model path) (encoded path)" << std::endl;
    exit(-1);
  }
  int buffer_size = STRTONUM<char*, int>(argv[1]);
  int num_block_data = STRTONUM<char*, int>(argv[2]);
  std::string load_path = argv[3];
  std::string encode_path = argv[4];
  BitEnsemble* encode = new BitEnsemble(buffer_size, num_block_data);
  BitEnsemble* decode = new BitEnsemble(buffer_size, num_block_data);
  bool pretty = false;
  if (pretty) {
    std::cout << "Load Model" << std::endl;
  }
  auto load_start = std::chrono::high_resolution_clock::now();
  encode->Load(load_path);
  auto load_end = std::chrono::high_resolution_clock::now();
  double load_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(load_end - load_start).count();
  if (pretty) {
    std::cout << "Start to Encode" << std::endl;
  }
  auto encode_start = std::chrono::high_resolution_clock::now();
  encode->Encode(encode_path);
  auto encode_end = std::chrono::high_resolution_clock::now();
  double encode_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(encode_end - encode_start).count();
  if (pretty) {
    std::cout << "Start to Decode" << std::endl;
  }
  auto decode_start = std::chrono::high_resolution_clock::now();
  decode->Decode(encode_path);
  auto decode_end = std::chrono::high_resolution_clock::now();
  double decode_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(decode_end - decode_start).count();
  if (pretty) {
    std::cout << "Load Time:\t" << load_dur / 1e9 << "s" << std::endl;
    std::cout << "Encode Time:\t" << encode_dur / 1e9 << "s" << std::endl;
    std::cout << "Decode Time:\t" << decode_dur / 1e9 << "s" << std::endl;
    std::cout << "Metadata Bits:\t" << encode->metadata_bits_ / 8. << "B" << std::endl;
    std::cout << "Pattern Bits:\t" << encode->pattern_bits_ / 8. << "B" << std::endl;
    std::cout << "Quantization Bits:\t" << encode->quantization_bits_ << "B" << std::endl;
  } else {
    std::cout << std::fixed << std::setprecision(10) << load_dur / 1e9 
              << "\t" << encode_dur / 1e9 << "\t" << decode_dur / 1e9 
              << "\t" << encode->metadata_bits_ / 8. << "\t" << encode->pattern_bits_ / 8.
              << "\t" << encode->quantization_bits_ / 8. << std::endl;
  }
  if (encode->Equal(decode)) {
    if (pretty) {
      std::cout << "Succeed to Encode/Decode" << std::endl;
    }
  } else {
    std::cout << "Fail to Encode/Decode" << std::endl;
  }
  return 0;
}