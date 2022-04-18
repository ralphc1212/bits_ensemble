#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <pthread.h>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

const long long kSEED = 1e9 + 7;
const int kINF = 0x7fffffff;

template<typename T>
inline bool EQ(const T& a, const T& b) {
  if (std::numeric_limits<T>::is_integer) {
    return a == b;
  } else {
    return std::fabs(a - b) < std::numeric_limits<T>::epsilon();
  }
}


template<typename T>
std::string STR(T n) {
  std::stringstream ss;
  ss << std::setprecision(std::numeric_limits<T>::digits10) << std::fixed << n;
  std::string n_str = ss.str();
  if (n_str.find(".") != std::string::npos) {
    while (*(n_str.rbegin()) == '0') {
      n_str.pop_back();
    }
    if (*(n_str.rbegin()) == '.') {
      n_str.pop_back();
    }
  }
  return n_str;
}

template<typename T, typename P>
P STRTONUM(T s) {
  std::string str = STR<T>(s);
  P v = 0;
  int point = -1;
  bool negative = (str[0] == '-');
  for (int i = (negative ? 1 : 0); i < str.size(); ++ i) {
    if (str[i] >= '0' && str[i] <= '9') {
      v = v * 10 + (str[i] - '0');
    } else if (point == -1 && str[i] == '.') {
      point = str.size() - i - 1;
    } else {
      std::cout << str + " is not a number" << std::endl;
      assert(false);
    }
  }
  for (int i = 0; i < point; ++ i) {
    v = v / 10.;
  }
  if (negative) {
    v = -v;
  }
  return v;
}

#endif