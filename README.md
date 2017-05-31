# cuYASHE

[University of Campinas](http://www.unicamp.br), [Institute of Computing](http://www.ic.unicamp.br), Brazil.

Laboratory of Security and Cryptography - [LASCA](http://www.lasca.ic.unicamp.br),<br>
Laboratório Multidisciplinar de Computação de Alto Desempenho - [LMCAD](http://www.lmcad.ic.unicamp.br). <br>

Author: [Pedro G. M. R. Alves](http://www.iampedro.com), PhD. student @ IC-UNICAMP,<br/>
Advisor: Professor [Diego F. Aranha](http://www.ic.unicamp.br/~dfaranha). <br/>

## About

cuYASHE is the first implementation of the leveled fully homomorphic scheme YASHE on GPGPUs. This library employs the CUDA platform and some algebric technics (like CRT, FFT and optimizations on polynomial and modular reduction)  to obtain significant performance improvements. When compared with the state-of-the-art implementation in CPU , GPU and FPGA, it shows speed-ups for all operations. In particular, there was an improvement between 6 and 35 times for polynomial multiplication.

## Goal

cuYASHE is an ongoing project and we hope to increase its performance and security in the course of time. Our focus is to provide:

 * Exceptional performance on modern GPGPUs.
 * A simple API, easy to use and very transparent.
 * Easily maintainable code. Easy to fix bugs and easy to scale.
 * A model for implementations of cryptographic schemes based on RLWE.
 
## Citing
If you use cuYASHE, please cite using the template below:

	@MastersThesis{Alves2016b,
		author     =     {Alves, Pedro G. M. R. and Aranha, Diego F.},
		title     =     {{Efficient GPGPU implementation of the Leveled Fully Homomorphic Encryption scheme YASHE}},
		school     =     {Institute of Computing, University of Campinas},
		address     =     {Brazil},
		year     =     {2016},
		note = 	{(In Portuguese)}
	}

## Licensing

cuYASHE is released under GPLv3.

## Disclaimer

cuYASHE is at most alpha-quality software. Implementations may not be correct or secure. Moreover, it was not tested with YASHE parameters different from those in the test file. This version should break with parameters much bigger than those. Use at your own risk.

## References

- Alves, P. G. M. R., & Aranha, D. F. (2016). Efficient GPGPU implementation of the Leveled Fully Homomorphic Encryption scheme YASHE (In Portuguese). Institute of Computing, Unicamp.
- Bos, J. W., Lauter, K., Loftus, J., & Naehrig, M. (n.d.). Improved Security for a Ring-Based Fully Homomorphic Encryption Scheme.


**Privacy Warning:** This site tracks visitor information.

