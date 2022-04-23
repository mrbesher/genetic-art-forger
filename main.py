#!/usr/bin/env python
# coding: utf-8
from sys import argv
from forger import Forger
from myopencvutil import plot_image, leave

if __name__ != '__main__':
    print('Run only as the main program')
    exit()

if len(argv) < 2:
    print('Please provide a file name')
    print('> ./forger filepath')
    exit()


filename = argv[1]

f = Forger(filename, population_size=1000, generation_num=1000, parent_num=50, dynamic_path=True)

f.fabricate(mu = 0.03)
