#!/usr/bin/python
#
# cuYASHE
# Copyright (C) 2015-2016 cuYASHE Authors
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from fractions import gcd
import prime as Prime

# Receives a integer and a list.
# Returns true if n is coprime with every integer in the list
def coprime_check(n, l):
	for i in l:
		if gcd(n,i) != 1:
			return False
	return True


# Generates a list the 200 biggest coprime integers lower than n bits 
coprimes = {}
for n in range(9,32):
	coprimes[n] = list()
	for i in xrange(pow(2,n)-1,0,-1):
		if coprime_check(i,coprimes[n]) is True:
			# Comment this to generate primes instead of coprimes
			if Prime.is_prime(i):
				coprimes[n].append(i)
			# coprimes[n].append(i)
		if len(coprimes[n]) >= 200:
			break

# Writes this to a .h file
f = open("coprimes.cpp","w+")
f.write("#include \"../settings.h\"")
# f.write("#ifndef PRIMES_H\n")
# f.write("#define PRIMES_H\n")
f.write("\n")
sizes = coprimes.keys()
sizes.sort()

def write_to_file(coprimes,sizes):
	prime_size = sizes.pop(0)

	f.write("#if CRTPRIMESIZE == " + str(prime_size) + "\n")
	f.write("\tconst uint32_t COPRIMES_BUCKET[] = { ")
	for i in coprimes[prime_size][1:]:
		if i != coprimes[prime_size][-1]:
			f.write(str(i) + ", ")
		else:
			f.write(str(i) + "};\n")
	f.write("#endif\n")
	if len(sizes) == 0:
		return
	else:
		write_to_file(coprimes,sizes)
		return

write_to_file(coprimes,sizes)
# f.write("\n#endif")
f.close()