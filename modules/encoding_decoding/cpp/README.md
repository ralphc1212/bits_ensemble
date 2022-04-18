# Input Formats
* $m\in\mathbb{N}$, the number of memebers; $h\in\mathbb{N}$, the number of weights; $s\in\mathbb{N}$, the number of steps; $b\in\mathbb{N}$, the number of bits each code requires.
* $w\in\mathbb{R}^{m\times h}$, quantized weights.
* $c\in\mathbb{N}^{m\times h\times (s - 1)}$, code book.

# Codebook
The number of codes is related to the number of memebers and is the sum of the second Stirling Numbers.

Take 4 members as an example, all 15 sharing schemes are,
```
{
  {00, 0000}, 
  {01, 0001}, 
  {02, 0010}, 
  {03, 0100}, 
  {04, 1000}, 
  {05, 0011}, 
  {06, 0110}, 
  {07, 1010}, 
  {08, 0012}, 
  {09, 0102}, 
  {10, 1002}, 
  {11, 0120},
  {12, 1020},
  {13, 1200},
  {14, 0123}
}
```
, where the number represents in which group the member is.

# Encoding

* For size numbers, store them directly.
* For quantized weights, store them in the bit array.
* For code book, store them in the bit array.