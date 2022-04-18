#ifndef BIT_ENSEMBLE_H
#define BIT_ENSEMBLE_H

#include "common.h"

#define BIT_TYPE uint8_t
#define BIT_SIZE (sizeof(BIT_TYPE) * 8)
#define BIT_LEN(n) static_cast<uint32_t>(std::ceil((n) * 1. / BIT_SIZE))

namespace bitensemble {

std::map<uint32_t, std::vector<std::vector<uint32_t>>> group_index = {{0, {{0, 1, 2, 3}}}, 
                                                                      {1, {{0, 1, 2}, {3}}},
                                                                      {2, {{0, 1, 3}, {2}}},
                                                                      {3, {{0, 2, 3}, {1}}},
                                                                      {4, {{1, 2, 3}, {0}}},
                                                                      {5, {{0, 1}, {2, 3}}},
                                                                      {6, {{0, 3}, {1, 2}}},
                                                                      {7, {{0, 2}, {1, 3}}},
                                                                      {8, {{0, 1}, {2}, {3}}},
                                                                      {9, {{0, 2}, {1}, {3}}},
                                                                      {10, {{1, 2}, {0}, {3}}},
                                                                      {11, {{0, 3}, {1}, {2}}},
                                                                      {12, {{1, 3}, {0}, {2}}},
                                                                      {13, {{2, 3}, {0}, {1}}},
                                                                      {14, {{0}, {1}, {2}, {3}}},
                                                                      };

class BitEnsemble {
public:
  double* weights_;
  uint32_t* quant_weights_;
  uint32_t quant_weights_bits_;
  uint32_t num_members_;
  uint32_t weight_dim_;
  double beta_;           // Max weight
  double alpha_;          // Min weight

  double* step_size_;
  uint32_t num_factors_;

  uint32_t* codes_;
  uint32_t bit_sharing_scheme_;

  BitEnsemble() : weights_(nullptr), quant_weights_(nullptr), step_size_(nullptr), codes_(nullptr) { }

  ~BitEnsemble() {
    if (weights_ != nullptr) {
      delete[] weights_;
    }
    if (quant_weights_ != nullptr) {
      delete[] quant_weights_;
    }
    if (step_size_ != nullptr) {
      delete[] step_size_;
    }
    if (codes_ != nullptr) {
      delete[] codes_;
    }
  }

  void Load(std::string path) {
    std::fstream in(path, std::ios::in);
    in >> num_members_ >> weight_dim_ >> num_factors_;
    assert(num_members_ == 4);
    weights_ = new double[num_members_ * weight_dim_];
    quant_weights_ = new uint32_t[num_members_ * weight_dim_ * num_factors_];
    step_size_ = new double[num_factors_];
    codes_ = new uint32_t[weight_dim_ * num_factors_];
    // Load the weights
    for (uint32_t i = 0; i < num_members_; ++ i) {
      for (uint32_t j = 0; j < weight_dim_; ++ j) {
        in >> weights_[i * weight_dim_ + j];
        if (i + j == 0) {
          beta_ = weights_[i * weight_dim_ + j];
          alpha_ = weights_[i * weight_dim_ + j];
        } else {
          beta_ = std::max(beta_, weights_[i * weight_dim_ + j]);
          alpha_ = std::min(alpha_, weights_[i * weight_dim_ + j]);
        }
      }
    }
    
    // Compute the step sizes
    for (uint32_t i = 0; i < num_factors_; ++ i) {
      if (i == 0) {
        step_size_[i] = (beta_ - alpha_) / 3;
      } else {
        step_size_[i] = step_size_[i - 1] / ((1 << (i + 1)) + 1);
      }
    }
    // Load the code book
    uint32_t max_code = 0;
    for (uint32_t j = 0; j < weight_dim_; ++ j) {
      for (uint32_t k = 1; k < num_factors_; ++ k) {
        in >> codes_[j * num_factors_ + k];
        max_code = std::max(max_code, codes_[j * num_factors_ + k]);
      }
    }
    for (bit_sharing_scheme_ = 0; max_code > 0; max_code >>= 1) {
      bit_sharing_scheme_ ++;
    }
    in.close();
    // Compute the quantized weights
    quant_weights_bits_ = 0;
    for (uint32_t j = 0; j < weight_dim_; ++ j) {
      for (uint32_t k = 0; k < num_factors_; ++ k) {
        if (k == 0) {
          for (uint32_t i = 0; i < num_members_; ++ i) {
            uint32_t idx = i * weight_dim_ * num_factors_ + j * num_factors_ + k;
            quant_weights_[idx] = std::round(weights_[i * weight_dim_ + j] / step_size_[k]);
            quant_weights_bits_ += 2;
            std::cout << "2bit\t" << i << "\t" << std::round(weights_[i * weight_dim_ + j] / step_size_[k]) << std::endl;
            assert((quant_weights_[idx] >> 2) == 0);
          }
        } else {
          uint32_t c = codes_[j * num_factors_ + k];
          uint32_t num_bits = (1 << k);
          const std::vector<std::vector<uint32_t>>& partitions = group_index[c];
          assert(partitions.size() > 0);
          for (uint32_t g = 0; g < partitions.size(); ++ g) {
            double res_err_avg = 0;
            for (uint32_t gi = 0; gi < partitions[g].size(); ++ gi) {
              uint32_t i = partitions[g][gi];
              uint32_t idx = i * weight_dim_ * num_factors_ + j * num_factors_ + k;
              double res_err = weights_[i * weight_dim_ + j] - quant_weights_[idx - 1] * step_size_[k - 1];
              std::cout << "each\t" << gi << "\t" << std::round(res_err / step_size_[k]) << std::endl;
              res_err_avg += res_err;
            }
            res_err_avg /= partitions[g].size();
            quant_weights_bits_ += num_bits;
            for (uint32_t gi = 0; gi < partitions[g].size(); ++ gi) {
              uint32_t i = partitions[g][gi];
              uint32_t idx = i * weight_dim_ * num_factors_ + j * num_factors_ + k;
              quant_weights_[idx] = std::round(res_err_avg / step_size_[k]);
              std::cout << "avg\t" << gi << "\t" << std::round(res_err_avg / step_size_[k]) << std::endl;
              assert((quant_weights_[idx] >> num_bits) == 0);
            }
          }
        }
      }
    }
    exit(-1);
  }

  bool Equal(const BitEnsemble* other) {
    if (quant_weights_bits_ != other->quant_weights_bits_ || 
        num_members_ != other->num_members_ ||
        weight_dim_ != other->weight_dim_ ||
        !EQ(beta_, other->beta_) ||
        !EQ(alpha_, other->alpha_) ||
        num_factors_ != other->num_factors_ ||
        bit_sharing_scheme_ != other->bit_sharing_scheme_) {
      std::cout << "This\nQuantized weight bits:\t" << quant_weights_bits_ 
                << "\n# Members:\t" << num_members_ << "\nWeight Dim:\t" 
                << weight_dim_ << "\nBeta:\t" << beta_ << "\nAlpha:\t" << alpha_
                << "\n# Factors:\t" << num_factors_ << "\nBit Sharing Scheme:\t" 
                << bit_sharing_scheme_ << std::endl;
      std::cout << "\nOther\nQuantized weight bits:\t" << other->quant_weights_bits_ 
                << "\n# Members:\t" << other->num_members_ << "\nWeight Dim:\t" 
                << other->weight_dim_ << "\nBeta:\t" << other->beta_ << "\nAlpha:\t" 
                << other->alpha_ << "\n# Factors:\t" << other->num_factors_ 
                << "\nBit Sharing Scheme:\t" << other->bit_sharing_scheme_ << std::endl;
      return false;
    }
    bool check = true;
    for (uint32_t i = 0; i < num_factors_; ++ i) {
      if (!EQ(step_size_[i], other->step_size_[i])) {
        check = false;
        break;
      }
    }
    if (!check) {
      std::cout << "Step sizes are different" << std::endl;
      return false;
    }
    check = true;
    for (uint32_t j = 0; j < weight_dim_; ++ j) {
      for (uint32_t k = 0; k < num_factors_; ++ k) {
        for (uint32_t i = 0; i < num_members_; ++ i) {
          uint32_t idx = i * weight_dim_ * num_factors_ + j * num_factors_ + k;
          if (quant_weights_[idx] != other->quant_weights_[idx]) {
            check = false;
            break;
          }
        }
      }
    }
    if (!check) {
      std::cout << "Quantized weights are different" << std::endl;
      return false;
    }
    return true;
  }
};

class EncodedBitEnsemble {
private:
  BIT_TYPE* quant_weights_;
  uint32_t quant_weights_bits_;
  uint32_t num_members_;
  uint32_t weight_dim_;
  double beta_;
  double alpha_;

  double* step_size_;
  uint32_t num_factors_;

  BIT_TYPE* codes_;
  uint32_t bit_sharing_scheme_;

public:
  EncodedBitEnsemble() : quant_weights_(nullptr), step_size_(nullptr), codes_(nullptr) { }

  ~EncodedBitEnsemble() {
    if (quant_weights_ != nullptr) {
      delete[] quant_weights_;
    }
    if (step_size_ != nullptr) {
      delete[] step_size_;
    }
    if (codes_ != nullptr) {
      delete[] codes_;
    }
  }

  uint32_t size() {
    return sizeof(EncodedBitEnsemble) + sizeof(double) * num_factors_ + 
            sizeof(BIT_TYPE) * (BIT_LEN(num_members_ * weight_dim_ * 
            bit_sharing_scheme_ * (num_factors_ - 1)) + BIT_LEN(quant_weights_bits_));
  }

  void Encode(BitEnsemble* bit_ensemble) {
    num_members_ = bit_ensemble->num_members_;
    weight_dim_ = bit_ensemble->weight_dim_;
    beta_ = bit_ensemble->beta_;
    alpha_ = bit_ensemble->alpha_;
    num_factors_ = bit_ensemble->num_factors_;
    step_size_ = new double[num_factors_];
    // Copy step size
    for (uint32_t i = 0; i < num_factors_; ++ i) {
      step_size_[i] = bit_ensemble->step_size_[i];
    }
    // Encode the code book
    bit_sharing_scheme_ = bit_ensemble->bit_sharing_scheme_;
    codes_ = new BIT_TYPE[BIT_LEN(weight_dim_ * bit_sharing_scheme_ * (num_factors_ - 1))];
    assert(bit_sharing_scheme_ <= BIT_SIZE);
    uint32_t u = 0;
    uint32_t cc = 0;
    uint32_t cur_bits = 0;
    for (uint32_t j = 0; j < weight_dim_; ++ j) {
      for (uint32_t k = 1; k < num_factors_; ++ k) {
        uint8_t c = bit_ensemble->codes_[j * num_factors_ + k];
        assert((c >> bit_sharing_scheme_) == 0);
        cc = (cc << bit_sharing_scheme_) | c;
        cur_bits += bit_sharing_scheme_;
        while (cur_bits >= BIT_SIZE) {
          uint32_t res_bits = cur_bits - BIT_SIZE;
          codes_[u ++] = (cc >> res_bits);
          cc &= ((1 << res_bits) - 1);
          cur_bits = res_bits;
        }
      }
    }
    while (cur_bits > 0) {
      uint32_t res_bits = cur_bits > BIT_SIZE ? cur_bits - BIT_SIZE : 0;
      codes_[u ++] = (cc >> res_bits);
      cc &= ((1 << res_bits) - 1);
      cur_bits = res_bits;
    }
    // Encode the weight matrix
    quant_weights_bits_ = bit_ensemble->quant_weights_bits_;
    quant_weights_ = new BIT_TYPE[BIT_LEN(quant_weights_bits_)];
    u = 0;
    cc = 0;
    cur_bits = 0;
    for (uint32_t j = 0; j < weight_dim_; ++ j) {
      for (uint32_t k = 0; k < num_factors_; ++ k) {
        if (k == 0) {
          for (uint32_t i = 0; i < num_members_; ++ i) {
            uint8_t c = bit_ensemble->quant_weights_[i * weight_dim_ * num_factors_ + j * num_factors_ + k];
            assert((c >> 2) == 0);
            cc = (cc << 2) | c;
            cur_bits += 2;
            while (cur_bits >= BIT_SIZE) {
              uint32_t res_bits = cur_bits - BIT_SIZE;
              quant_weights_[u ++] = (cc >> res_bits);
              cc &= ((1 << res_bits) - 1);
              cur_bits = res_bits;
            }
          }
        } else {
          uint32_t c = bit_ensemble->codes_[j * num_factors_ + k];
          uint32_t num_bits = (1 << k);
          const std::vector<std::vector<uint32_t>>& partitions = group_index[c];
          for (uint32_t g = 0; g < partitions.size(); ++ g) {
            uint32_t i = partitions[g][0];
            uint8_t c = bit_ensemble->quant_weights_[i * weight_dim_ * num_factors_ + j * num_factors_ + k];
            assert((c >> num_bits) == 0);
            cc = (cc << num_bits) | c;
            cur_bits += num_bits;
            while (cur_bits >= BIT_SIZE) {
              uint32_t res_bits = cur_bits - BIT_SIZE;
              quant_weights_[u ++] = (cc >> res_bits);
              cc &= ((1 << res_bits) - 1);
              cur_bits = res_bits;
            }
          }
        }
      }
    }
    while (cur_bits > 0) {
      uint32_t res_bits = cur_bits > BIT_SIZE ? cur_bits - BIT_SIZE : 0;
      codes_[u ++] = (cc >> res_bits);
      cc &= ((1 << res_bits) - 1);
      cur_bits = res_bits;
    }
  }

  BitEnsemble* Decode() {
    BitEnsemble* bit_ensemble = new BitEnsemble();
    bit_ensemble->quant_weights_bits_ = quant_weights_bits_;
    bit_ensemble->num_members_ = num_members_;
    bit_ensemble->weight_dim_ = weight_dim_;
    bit_ensemble->beta_ = beta_;
    bit_ensemble->alpha_ = alpha_;
    bit_ensemble->num_factors_ = num_factors_;
    bit_ensemble->step_size_ = new double[num_factors_];
    // Copy step size
    for (uint32_t i = 0; i < num_factors_; ++ i) {
      bit_ensemble->step_size_[i] = step_size_[i];
    }
    // Decode the code book
    bit_ensemble->bit_sharing_scheme_ = bit_sharing_scheme_;
    bit_ensemble->codes_ = new uint32_t[weight_dim_ * num_factors_];
    assert(bit_sharing_scheme_ <= BIT_SIZE);
    uint32_t u = 0;
    uint32_t cc = 0;
    uint32_t cur_bits = 0;
    for (uint32_t j = 0; j < weight_dim_; ++ j) {
      for (uint32_t k = 1; k < num_factors_; ++ k) {          
        while (cur_bits < bit_sharing_scheme_) {
          cc = (cc << BIT_SIZE) | codes_[u ++];
          cur_bits += BIT_SIZE;
        }
        uint32_t res_bits = cur_bits - bit_sharing_scheme_;
        uint32_t c = (cc >> res_bits);
        cc &= ((1 << res_bits) - 1);
        cur_bits = res_bits;
        assert((c >> bit_sharing_scheme_) == 0);
        bit_ensemble->codes_[j * num_factors_ + k] = c;
      }
    }
    // Decode the weight matrix
    bit_ensemble->quant_weights_ = new uint32_t[num_members_ * weight_dim_ * num_factors_];
    u = 0;
    cc = 0;
    cur_bits = 0;
    for (uint32_t j = 0; j < weight_dim_; ++ j) {
      for (uint32_t k = 0; k < num_factors_; ++ k) {
        if (k == 0) {
          for (uint32_t i = 0; i < num_members_; ++ i) {
            while (cur_bits < 2) {
              cc = (cc << BIT_SIZE) | quant_weights_[u ++];
              cur_bits += BIT_SIZE;
            }
            uint32_t res_bits = cur_bits - 2;
            uint8_t c = (cc >> res_bits);
            cc &= ((1 << res_bits) - 1);
            cur_bits = res_bits;
            assert((c >> 2) == 0);
            bit_ensemble->quant_weights_[i * weight_dim_ * num_factors_ + j * num_factors_ + k] = c;
          }
        } else {
          uint32_t c = bit_ensemble->codes_[j * num_factors_ + k];
          uint32_t num_bits = (1 << k);
          const std::vector<std::vector<uint32_t>>& partitions = group_index[c];
          for (uint32_t g = 0; g < partitions.size(); ++ g) {
            while (cur_bits < num_bits) {
              cc = (cc << BIT_SIZE) | quant_weights_[u ++];
              cur_bits += BIT_SIZE;
            }
            uint32_t res_bits = cur_bits - BIT_SIZE;
            uint32_t c = (cc >> res_bits);
            cc &= ((1 << res_bits) - 1);
            cur_bits = res_bits;
            assert((c >> num_bits) == 0);
            for (uint32_t gi = 0; gi < partitions[g].size(); ++ gi) {
              uint32_t i = partitions[g][gi];
              bit_ensemble->quant_weights_[i * weight_dim_ * num_factors_ + j * num_factors_ + k] = c;
            }
          }
        }
      }
    }
    return bit_ensemble;
  }
};

}

#endif
