# cuYASHE

University of Campinas, Institute of Computing, Brazil.

Laboratory of Security and Cryptography - [LASCA](http://www.lasca.ic.unicamp.br),<br>
Laboratório Multidisciplinar de Computação de Alto Desempenho - [LMCAD](http://www.lmcad.ic.unicamp.br). <br>

Author: [Pedro G. M. R. Alves](http://www.iampedro.com), PhD. student @ IC-UNICAMP,<br/>
Advisor: [Diego F. Aranha](http://www.ic.unicamp.br/~dfaranha). <br/>

## About

cuYASHE is a library that implements the leveled fully homomorphic scheme YASHE on GPGPUs. The implementation employs the CUDA platform and some algebric technics (like CRT, FFT and optimizations on polynomial and modular reduction)  to obtain significant performance improvements. When compared with the state-of-the-art implementation in CPU , GPU and FPGA, it shows speed-ups for all operations. In particular, there was an improvement between 6 and 35 times for polynomial multiplication.

## Goal

cuYASHE is an ongoing project and we hope to increase its performance and security in the course of time. Our focus is to provide:

 * Exceptional performance on modern GPGPUs.
 * A simple API, easy to use and very transparent.
 * Easily maintainable code. Easy to fix bugs and easy to scale.
 * A model for implementations of cryptographic schemes based on RLWE.
 
## Citing
If you use cuYASHE, please cite using the template below:

	@msc dissertation{Alves2016b,
 		author = "P. Alves and D. Aranha",
  		title = "Computação sobre dados cifrados em GPGPUs",
  		year = 2016,
  		publisher = "UNICAMP",
  		institution = "Institute of Computing",
  		month = "Jun",
	        howpublished = {\url{https://github.com/cuyashe-library/cuyashe}}
	}


# Licensing

cuYASHE is released under an GPLv3 license.

## Disclaimer

cuYASHE is at most alpha-quality software. Implementations may not be correct or secure. Use at your own risk.

## References

- Alves, P. G. M. R., & Aranha, D. F. (2016). Computação sobre dados cifrados em GPGPUs. Institute of Computing, Unicamp.
- Bos, J. W., Lauter, K., Loftus, J., & Naehrig, M. (n.d.). Improved Security for a Ring-Based Fully Homomorphic Encryption Scheme.


**Privacy Warning:** This site tracks visitor information.

