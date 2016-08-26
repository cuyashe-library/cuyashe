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
import sys
from multiprocessing import Pool
import os

# Receives a integer and a list.
# Returns true if n is coprime with every integer in the list
def coprime_check(n, l):
	for i in l:
		if gcd(n,i) != 1:
			return False
	return True

def is_int(x):
	try:
		int(x) == x
		return True
	except Exception, e:
		return False

def worker(data):
	COPRIMES_BUCKET_SIZE = data[0]
	j = data[1]
	print "%d - %d " % (j,COPRIMES_BUCKET_SIZE)
	coprimes = {}
	coprimes[j] = list()

	for i in xrange(pow(2,j)-1,0,-1):
		if coprime_check(i,coprimes[j]) is True:
		# Comment this to generate primes instead of coprimes
		# if Prime.is_prime(i):
			coprimes[j].append(i)
		if len(coprimes[j]) >= COPRIMES_BUCKET_SIZE:
			break
	return coprimes

def compute_coprimes(COPRIMES_BUCKET_SIZE):

	# Generates a list the 200 biggest coprime integers lower than n bits 
	coprimes = {}
	prime_range = range(9,31)

	p = Pool()
	z = [COPRIMES_BUCKET_SIZE]*len(prime_range)
	m = p.map_async(worker, zip(z,prime_range))
	
	while not m.ready():
		pass

	result = m.get()

	for c in result:
		coprimes.update(c)
	return coprimes

def write_to_file(coprimes,sizes):
	prime_size = sizes.pop(0)
	f.write("#if CRTPRIMESIZE == " + str(prime_size) + "\n")
	# f.write("\t#define COPRIMES_BUCKET_SIZE " + str(len(coprimes[prime_size])) + "\n")
	f.write("\tconst int COPRIMES_BUCKET[] = { ")
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

def write_to_header(coprimes,sizes):
	while len(sizes) > 0:
		prime_size = sizes.pop(0)

		f.write("#if CRTPRIMESIZE == " + str(prime_size) + "\n")
		f.write("\t#define COPRIMES_BUCKET_SIZE " + str(len(coprimes[prime_size])) + "\n")
		f.write("\textern const int COPRIMES_BUCKET[COPRIMES_BUCKET_SIZE];\n")
		f.write("#endif\n")

if __name__ == "__main__":
	COPRIMES_BUCKET_SIZE = 200

	if len(sys.argv) > 1:
		COPRIMES_BUCKET_SIZE = int(sys.argv[1])

	print "Setting COPRIMES_BUCKET_SIZE to %s\n" % COPRIMES_BUCKET_SIZE

	# Generates a list the 200 biggest coprime integers lower than n bits 
	coprimes = compute_coprimes(COPRIMES_BUCKET_SIZE)

	# f.write("#ifndef PRIMES_H\n")
	# f.write("#define PRIMES_H\n")
	sizes = list(coprimes.keys())
	sizes.sort()
	
	# Writes this to a .cpp file
	f = open("coprimes.cpp","w+")

	f.write("#include \"coprimes.h\"\n")
	f.write("\n")
	write_to_file(coprimes,sizes)
	# f.write("\n#endif")
	f.close()

	sizes = list(coprimes.keys())
	sizes.sort()

	# Generates the header
	f = open("coprimes.h","w+")
	f.write("#ifndef COPRIMES_H\n")
	f.write("#define COPRIMES_H\n")
	f.write("#include \"../settings.h\"\n")
	write_to_header(coprimes,sizes)
	f.write("#endif")

