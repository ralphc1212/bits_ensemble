#ifndef BIT_ENSEMBLE_H
#define BIT_ENSEMBLE_H

#include "common.h"

#define BIT_TYPE uint8_t
#define BIT_SIZE (sizeof(BIT_TYPE) * 8)
#define LOW_BIT(x, n) (x & ((static_cast<uint32_t>(1) << (n)) - 1))

std::map<uint32_t, std::vector<std::vector<uint32_t>>> group_index = {{0, {{0, 1, 2, 3}}}, 
                                                                      {1, {{0, 1, 2}, {3}}},
                                                                      {2, {{0, 1, 3}, {2}}},
                                                                      {3, {{0, 2, 3}, {1}}},
                                                                      {4, {{1, 2, 3}, {0}}},
                                                                      {5, {{0, 1}, {2, 3}}},
                                                                      {6, {{0, 2}, {1, 3}}},
                                                                      {7, {{0, 3}, {1, 2}}},
                                                                      {8, {{0, 1}, {2}, {3}}},
                                                                      {9, {{0, 2}, {1}, {3}}},
                                                                      {10, {{0, 3}, {1}, {2}}},
                                                                      {11, {{1, 2}, {0}, {3}}},
                                                                      {12, {{1, 3}, {0}, {2}}},
                                                                      {13, {{2, 3}, {0}, {1}}},
                                                                      {14, {{0}, {1}, {2}, {3}}},
                                                                      };

class BitEnsemble {
private:
  uint32_t step_size_;
  uint32_t num_members_;
  uint32_t weight_dim_;
  uint32_t** max_step_;
  std::vector<uint32_t**> quant_weights_;
  uint32_t** codes_;

  // std::vector<BIT_TYPE> bit_buf_;
  BIT_TYPE* bit_buf_;
  uint32_t buf_idx_;
  uint32_t cur_bits_;
  uint32_t kBufferSize;
  uint32_t kNumBlockData;

public:
  uint32_t metadata_bits_;
  uint32_t pattern_bits_;
  uint32_t quantization_bits_;

  BitEnsemble(uint32_t buffer_size, uint32_t num_block_data) : max_step_(nullptr), codes_(nullptr), cur_bits_(0), buf_idx_(0), 
                  kBufferSize(buffer_size), kNumBlockData(num_block_data) {
    // bit_buf_.resize(kBufferSize);
    bit_buf_ = new BIT_TYPE[buffer_size];
    memset(bit_buf_, 0, sizeof(BIT_TYPE) * buffer_size);
    metadata_bits_ = 0;
    pattern_bits_ = 0;
    quantization_bits_ = 0;
  }

  ~BitEnsemble() {
    if (bit_buf_ != nullptr) {
      delete[] bit_buf_;
    }
    destory();
  }

  void Load(std::string path, bool auto_correct=true) {
    std::fstream in(path, std::ios::in);
    if (!in.is_open()) {
      std::cout << "Fail to open [" << path << "]" << std::endl;
      exit(-1);
    }
    in >> step_size_ >> weight_dim_ >> num_members_;
    assert(num_members_ == 4);
    init();
    // Load the # steps for each weight
    for (uint32_t i = 0; i < weight_dim_; ++ i) {
      for (uint32_t j = 0; j < num_members_; ++ j) {
        in >> max_step_[i][j];
      }
    }
    // Load the code book
    for (uint32_t i = 0; i < weight_dim_; ++ i) {
      for (uint32_t s = 0; s < step_size_ - 1; ++ s) {
        in >> codes_[i][s];
      }
    }
    // Load the quantized weights
    for (uint32_t i = 0; i < weight_dim_; ++ i) {
      for (uint32_t j = 0; j < num_members_; ++ j) {
        for (uint32_t s = 0; s < step_size_; ++ s) {
          uint32_t num_bits = (s == 0 ? 2 : (1 << s));
          int weight = 0;
          in >> weight;
          if (weight < 0) {
            weight += (1 << num_bits) / 2;
          } else if (weight > 0) {
            weight += ((1 << num_bits) / 2 - 1);
          }
          assert(weight >= 0);
          assert(weight < (1 << num_bits));
          quant_weights_[i][j][s] = weight;
        }
      }
    }
    // Check quantized weights
    std::set<uint32_t> mems;
    for (uint32_t i = 0; i < weight_dim_; ++ i) {
      for (uint32_t s = 1; s < step_size_; ++ s) {
        uint32_t c = codes_[i][s - 1];
        mems.clear();
        for (uint32_t j = 0; j < num_members_; ++ j) {
          if (max_step_[i][j] >= s) {
            mems.insert(j);
          } else {
            if (auto_correct) {
              quant_weights_[i][j][s] = 0;
            }
          }
        }
        const std::vector<std::vector<uint32_t>>& partitions = group_index[c];
        uint32_t num_bits = (1 << s);
        for (uint32_t g = 0; g < partitions.size(); ++ g) {
          uint32_t j0;
          bool has = false;
          for (uint32_t gi = 0; gi < partitions[g].size(); ++ gi) {
            uint32_t j = partitions[g][gi];
            if (mems.find(partitions[g][gi]) != mems.end()) {
              if (!has) {
                j0 = partitions[g][gi];
                has = true;
              } else {
                if (quant_weights_[i][j0][s] != quant_weights_[i][j][s]) {
                  // std::cout << "i [" << i << "], s [" << s << "]" << std::endl;
                  // std::cout << j0 << "th Member:" << quant_weights_[i][j0][s] << std::endl;
                  // std::cout << j << "th Member:" << quant_weights_[i][j][s] << std::endl;
                  // exit(-1);
                  if (auto_correct) {
                    quant_weights_[i][j][s] = quant_weights_[i][j0][s];
                  }
                }
                assert(quant_weights_[i][j0][s] == quant_weights_[i][j][s]);
              }
            }
          }
        }
      }
    }
    in.close();
  }

  void Encode(std::string path) {
    std::ofstream out(path, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
      std::cout << "Fail to open [" << path << "]" << std::endl;
      exit(-1);
    }
    write(step_size_, 2, out);
    write(weight_dim_, sizeof(uint32_t) * 8, out);
    write(num_members_, sizeof(uint32_t) * 8, out);
    metadata_bits_ += 2 + sizeof(uint32_t) * 8 * 2;
    assert(kNumBlockData % num_members_ == 0);
    uint32_t num_weight_block = kNumBlockData / num_members_;
    for (uint32_t i = 0; i < weight_dim_; i += num_weight_block) {
      uint32_t l = i;
      uint32_t r = std::min(weight_dim_, i + num_weight_block);
      EncodeBlock(l, r, out);
    }
    flush(out);
    out.close();
  }

  void Decode(std::string path) {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
      std::cout << "Fail to open [" << path << "]" << std::endl;
      exit(-1);
    }
    memset(bit_buf_, 0, sizeof(BIT_TYPE) * kBufferSize);
    in.read(reinterpret_cast<char*>(bit_buf_), sizeof(BIT_TYPE) * kBufferSize);
    buf_idx_ = 0;
    read(step_size_, 2, in);
    read(weight_dim_, sizeof(uint32_t) * 8, in);
    read(num_members_, sizeof(uint32_t) * 8, in);
    init();
    assert(kNumBlockData % num_members_ == 0);
    uint32_t num_weight_block = kNumBlockData / num_members_;
    for (uint32_t i = 0; i < weight_dim_; i += num_weight_block) {
      uint32_t l = i;
      uint32_t r = std::min(weight_dim_, i + num_weight_block);
      DecodeBlock(l, r, in);
    }
    in.close();
  }

  bool Equal(const BitEnsemble* other) {
    if (step_size_ != other->step_size_) {
      std::cout << "Step Size Error" << std::endl;
      std::cout << "True [" << step_size_ << "]" << std::endl;
      std::cout << "False [" << other->step_size_ << "]" << std::endl;
      return false;
    }
    if (weight_dim_ != other->weight_dim_) {
      std::cout << "Weight Dim Error" << std::endl;
      std::cout << "True [" << weight_dim_ << "]" << std::endl;
      std::cout << "False [" << other->weight_dim_ << "]" << std::endl;
      return false;
    }
    if (num_members_ != other->num_members_) {
      std::cout << "# Members Error" << std::endl;
      std::cout << "True [" << num_members_ << "]" << std::endl;
      std::cout << "False [" << other->num_members_ << "]" << std::endl;
      return false;
    }
    for (uint32_t i = 0; i < weight_dim_; ++ i) {
      for (uint32_t j = 0; j < num_members_; ++ j) {
        if (max_step_[i][j] != other->max_step_[i][j]) {
          std::cout << "Max Step Error [" << i << ", " << j << "]" << std::endl;
          std::cout << "True [" << max_step_[i][j] << "]" << std::endl;
          std::cout << "False [" << other->max_step_[i][j] << "]" << std::endl;
          return false;
        }
      }
    }
    for (uint32_t i = 0; i < weight_dim_; ++ i) {
      for (uint32_t s = 0; s < step_size_ - 1; ++ s) {
        if (codes_[i][s] != other->codes_[i][s]) {
          std::cout << "Code Book Error [" << i << ", " << s << "]" << std::endl;
          std::cout << "True [" << codes_[i][s] << "]" << std::endl;
          std::cout << "False [" << other->codes_[i][s] << "]" << std::endl;
          return false;          
        }
      }
    }
    for (uint32_t i = 0; i < weight_dim_; ++ i) {
      for (uint32_t j = 0; j < num_members_; ++ j) {
        for (uint32_t s = 0; s < step_size_; ++ s) {
          if (quant_weights_[i][j][s] != other->quant_weights_[i][j][s]) {
            std::cout << "Quantized Weight Error [" << i << ", " << j << ", " << s << "]" << std::endl;
            std::cout << "True [" << quant_weights_[i][j][s] << "]" << std::endl;
            std::cout << "False [" << other->quant_weights_[i][j][s] << "]" << std::endl;
            return false;
          }
        }
      }
    }
    return true;
  }

private:
  std::string binary(uint32_t x, uint32_t num_bits) {
    std::string s = "";
    for (uint32_t i = 0; i < num_bits; ++ i) {
      if (x & 1) {
        s = "1" + s;
      } else {
        s = "0" + s;
      }
      x >>= 1;
    }
    return s;
  }

  void init() {
    destory();
    max_step_ = new uint32_t*[weight_dim_];
    for (uint32_t i = 0; i < weight_dim_; ++ i) {
      max_step_[i] = new uint32_t[num_members_];
      memset(max_step_[i], 0, sizeof(uint32_t) * num_members_);
    }
    quant_weights_.resize(weight_dim_);
    codes_ = new uint32_t*[weight_dim_];
    for (uint32_t i = 0; i < weight_dim_; ++ i) {
      codes_[i] = new uint32_t[step_size_ - 1];
      memset(codes_[i], 0, sizeof(uint32_t) * (step_size_ - 1));
    }
    for (uint32_t i = 0; i < weight_dim_; ++ i) {
      quant_weights_[i] = new uint32_t*[num_members_];
      for (uint32_t j = 0; j < num_members_; ++ j) {
        quant_weights_[i][j] = new uint32_t[step_size_];
        memset(quant_weights_[i][j], 0, sizeof(uint32_t) * step_size_);
      }
    }
  }

  void destory() {
    if (max_step_ != nullptr) {
      for (uint32_t i = 0; i < weight_dim_; ++ i) {
        delete[] max_step_[i];
      }
      delete[] max_step_;
    }
    for (uint32_t i = 0; i < quant_weights_.size(); ++ i) {
      if (quant_weights_[i] != nullptr) {
        for (uint32_t j = 0; j < num_members_; ++ j) {
          delete[] quant_weights_[i][j];
        }
        delete[] quant_weights_[i];
      }
    }
    if (codes_ != nullptr) {
      for (uint32_t i = 0; i < weight_dim_; ++ i) {
        delete[] codes_[i];
      }
      delete[] codes_;
    }
  }

  void read(uint32_t& data, uint32_t bits, std::ifstream& in) {
    // std::cout << "\nStart to read [" << bits << "] bits data" << std::endl;
    // std::cout << "Start from [" << buf_idx_ << "] Buffer, Cur Bits [" << cur_bits_ << "]" << std::endl;
    data = 0;
    uint32_t res_bits = bits;
    if (res_bits >= BIT_SIZE - cur_bits_) {
      if (buf_idx_ >= kBufferSize) {
        memset(bit_buf_, 0, sizeof(BIT_TYPE) * kBufferSize);
        // std::cout << "Read Up 1" << std::endl;
        in.read(reinterpret_cast<char*>(bit_buf_), sizeof(BIT_TYPE) * kBufferSize);
        buf_idx_ = 0;
      }
      res_bits = res_bits - BIT_SIZE + cur_bits_;
      // std::cout << "Read the " << buf_idx_ << " slot with residual bits [" << BIT_SIZE - cur_bits_ << "]" << std::endl;
      data |= (LOW_BIT(bit_buf_[buf_idx_ ++],  BIT_SIZE - cur_bits_) << res_bits);
      cur_bits_ = 0;
      // std::cout << "Data [" << binary(data, bits) << "]" << std::endl;
    }
    // std::cout << "1-Rest bits [" << res_bits << "] to read" << std::endl;
    while (res_bits >= BIT_SIZE) {
      if (buf_idx_ >= kBufferSize) {
        memset(bit_buf_, 0, sizeof(BIT_TYPE) * kBufferSize);
        // std::cout << "Read Up 2" << std::endl;
        in.read(reinterpret_cast<char*>(bit_buf_), sizeof(BIT_TYPE) * kBufferSize);
        buf_idx_ = 0;
      }
      // std::cout << "Read the " << buf_idx_ << " slot" << std::endl;
      res_bits -= BIT_SIZE;
      data |= (bit_buf_[buf_idx_ ++] << res_bits);
      // std::cout << "Rest bits [" << res_bits << "] to read" << std::endl;
    }
    // std::cout << "2-Rest bits [" << res_bits << "] to read" << std::endl;
    if (res_bits > 0) {
      if (buf_idx_ >= kBufferSize) {
        memset(bit_buf_, 0, sizeof(BIT_TYPE) * kBufferSize);
        // std::cout << "Read Up 3" << std::endl;
        in.read(reinterpret_cast<char*>(bit_buf_), sizeof(BIT_TYPE) * kBufferSize);
        buf_idx_ = 0;
      }
      data |= LOW_BIT((bit_buf_[buf_idx_] >> (BIT_SIZE - cur_bits_ - res_bits)), res_bits);
      cur_bits_ += res_bits;
      if (cur_bits_ == BIT_SIZE) {
        buf_idx_ ++;
        cur_bits_ = 0;
      }
      // std::cout << "Read " << cur_bits_ << " bits in the " << buf_idx_ << " slot" << std::endl;
    }
    // std::cout << "Data [" << data << "] [" << binary(data, bits) << "]" << std::endl;
    // std::cout << "End to Read" << std::endl;
  }

  void write(uint32_t data, uint32_t bits, std::ofstream& out) {
    // std::cout << "\nStart to write data [" << data << "], bits [" << bits << "]" << std::endl;
    uint32_t res_bits = bits;
    if (res_bits >= BIT_SIZE - cur_bits_) { 
      // std::cout << "Write the " << buf_idx_ << " slot with residual bits [" << BIT_SIZE - cur_bits_ << "]" << std::endl;
      if (buf_idx_ >= kBufferSize) {
        // std::cout << "Write Down 1" << std::endl;
        out.write(reinterpret_cast<const char*>(bit_buf_), sizeof(BIT_TYPE) * kBufferSize);
        memset(bit_buf_, 0, sizeof(BIT_TYPE) * kBufferSize);
        buf_idx_ = 0;
      }
      res_bits = res_bits - BIT_SIZE + cur_bits_;
      bit_buf_[buf_idx_ ++] |= LOW_BIT((data >> res_bits), (BIT_SIZE - cur_bits_));
      cur_bits_ = 0;
    }
    // std::cout << "1-Rest bits [" << res_bits << "] to write" << std::endl;
    while (res_bits >= BIT_SIZE) {
      if (buf_idx_ >= kBufferSize) {
        // std::cout << "Write Down 2" << std::endl;
        out.write(reinterpret_cast<const char*>(bit_buf_), sizeof(BIT_TYPE) * kBufferSize);
        memset(bit_buf_, 0, sizeof(BIT_TYPE) * kBufferSize);
        buf_idx_ = 0;
      }
      // std::cout << "Write the " << buf_idx_ << " slot" << std::endl;
      res_bits -= BIT_SIZE;
      bit_buf_[buf_idx_ ++] = LOW_BIT((data >> res_bits), BIT_SIZE);
      // std::cout << "Rest bits [" << res_bits << "] to write" << std::endl;
    }
    // std::cout << "2-Rest bits [" << res_bits << "] to write" << std::endl;
    if (res_bits > 0) {
      if (buf_idx_ >= kBufferSize) {
        // std::cout << "Write Down 3" << std::endl;
        out.write(reinterpret_cast<const char*>(bit_buf_), sizeof(BIT_TYPE) * kBufferSize);
        memset(bit_buf_, 0, sizeof(BIT_TYPE) * kBufferSize);
        buf_idx_ = 0;
      }
      uint32_t emp_bits = BIT_SIZE - cur_bits_ - res_bits;
      bit_buf_[buf_idx_] = (((bit_buf_[buf_idx_] >> emp_bits) | LOW_BIT(data, res_bits)) << emp_bits);
      cur_bits_ += res_bits;
      if (cur_bits_ == BIT_SIZE) {
        buf_idx_ ++;
        cur_bits_ = 0;
      }
      // std::cout << "Write " << cur_bits_ << " bits in the " << buf_idx_ << " slot" << std::endl;
    }
    // std::cout << "End to Write" << std::endl;
  }

  void flush(std::ofstream& out) {
    if (buf_idx_ > 0 || cur_bits_ > 0) {
      // std::cout << "Write Down 4" << std::endl;
      out.write(reinterpret_cast<const char*>(bit_buf_), sizeof(BIT_TYPE) * (buf_idx_ + 1));
      memset(bit_buf_, 0, sizeof(BIT_TYPE) * kBufferSize);
    }
    buf_idx_ = 0;
    cur_bits_ = 0;
  }

  void EncodeBlock(uint32_t l, uint32_t r, std::ofstream& out) {
    // Encode max step
    std::vector<std::pair<uint32_t, uint32_t>> step_count;
    uint32_t step = 0;
    uint32_t ct = 0;
    for (uint32_t i = l; i < r; ++ i) {
      for (uint32_t j = 0; j < num_members_; ++ j) {
        if (max_step_[i][j] == step) {
          ct ++;
        } else {
          if (ct > 0) {
            step_count.push_back({step, ct - 1});
          }
          step = max_step_[i][j];
          ct = 1;
        }
      }
    }
    if (ct > 0) {
      step_count.push_back({step, ct - 1});
    }
    write(step_count.size(), sizeof(uint32_t) * 8, out);
    metadata_bits_ += sizeof(uint32_t) * 8;
    for (uint32_t b = 0; b < step_count.size(); ++ b) {
      // TODO: NOT use the const value
      write(step_count[b].first, 2, out);
      write(step_count[b].second, 8, out);
      metadata_bits_ += 10;
    }
    // Encode codes
    for (uint32_t i = l; i < r; ++ i) {
      for (uint32_t s = 0; s < step_size_ - 1; ++ s) {
        // TODO: NOT use the const value
        write(codes_[i][s], 4, out);
        pattern_bits_ += 4;
      }
    }
    // Encode quantized weights
    std::set<uint32_t> mems;
    for (uint32_t i = l; i < r; ++ i) {
      for (uint32_t s = 0; s < step_size_; ++ s) {
        if (s == 0) {
          for (uint32_t j = 0; j < num_members_; ++ j) {
            // TODO: NOT use the const value
            write(quant_weights_[i][j][s], 2, out);
            quantization_bits_ += 2;
          }
        } else {
          uint32_t c = codes_[i][s - 1];
          mems.clear();
          for (uint32_t j = 0; j < num_members_; ++ j) {
            if (max_step_[i][j] >= s) {
              mems.insert(j);
            }
          }
          const std::vector<std::vector<uint32_t>>& partitions = group_index[c];
          uint32_t num_bits = (1 << s);
          for (uint32_t g = 0; g < partitions.size(); ++ g) {
            uint32_t j;
            bool has = false;
            for (uint32_t gi = 0; gi < partitions[g].size(); ++ gi) {
              if (mems.find(partitions[g][gi]) != mems.end()) {
                j = partitions[g][gi];
                has = true;
                break;
              }
            }
            if (has) {
              write(quant_weights_[i][j][s], num_bits, out);
              quantization_bits_ += num_bits;
            }
          }
        }
      }
    }
  }

  void DecodeBlock(uint32_t l, uint32_t r, std::ifstream& in) {
    // Decode max step
    std::vector<std::pair<uint32_t, uint32_t>> step_count;
    uint32_t size;
    read(size, sizeof(uint32_t) * 8, in);
    for (uint32_t b = 0, i = l, j = 0; b < size; ++ b) {
      uint32_t step, ct;
      // TODO: NOT use the const value
      read(step, 2, in);
      read(ct, 8, in);
      for (uint32_t k = 0; k < ct + 1; ++ k) {
        assert(i < r);
        assert(i < weight_dim_);
        assert(j < num_members_);
        max_step_[i][j] = step;
        j ++;
        if (j >= num_members_) {
          i ++;
          j = j % num_members_;
        }
      }
    }
    // Decode code
    for (uint32_t i = l; i < r; ++ i) {
      for (uint32_t s = 0; s < step_size_ - 1; ++ s) {
        // TODO: NOT use the const value
        read(codes_[i][s], 4, in);
      }
    }
    // Decode quantized weight
    std::set<uint32_t> mems;
    for (uint32_t i = l; i < r; ++ i) {
      for (uint32_t s = 0; s < step_size_; ++ s) {
        if (s == 0) {
          for (uint32_t j = 0; j < num_members_; ++ j) {
            // TODO: NOT use the const value
            read(quant_weights_[i][j][s], 2, in);
          }
        } else {
          uint32_t c = codes_[i][s - 1];
          mems.clear();
          for (uint32_t j = 0; j < num_members_; ++ j) {
            if (max_step_[i][j] >= s) {
              mems.insert(j);
            }
          }
          const std::vector<std::vector<uint32_t>>& partitions = group_index[c];
          uint32_t num_bits = (1 << s);
          for (uint32_t g = 0; g < partitions.size(); ++ g) {
            bool has = false;
            for (uint32_t gi = 0; gi < partitions[g].size(); ++ gi) {
              if (mems.find(partitions[g][gi]) != mems.end()) {
                has = true;
                break;
              }
            }
            if (has) {
              uint32_t w;
              read(w, num_bits, in);
              for (uint32_t gi = 0; gi < partitions[g].size(); ++ gi) {
                uint32_t j = partitions[g][gi];
                if (mems.find(j) != mems.end()) {
                  quant_weights_[i][j][s] = w;
                }
              }
            }
          }
        }
      }
    }
  }
};

#endif
