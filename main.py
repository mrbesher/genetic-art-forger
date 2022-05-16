#!/usr/bin/env python
# coding: utf-8
import argparse
from pathlib import Path, PurePath
import matplotlib.pyplot as plt
from forger import Forger

parser = argparse.ArgumentParser()

parser.add_argument("imagepath", help="Path to the input image")
parser.add_argument("--dpath", action="store_true", help = "Increase path length dynamically")
parser.add_argument("--dmu", action="store_true", help = "Decrease mutation rate dynamically")
parser.add_argument("-p", action = "store_true", help = "Choose the best parents based on probabity")
parser.add_argument("--popsize", default = 1000, help = "Set the population size", type=int)
parser.add_argument("--gennum", default = 1000, help = "Set the number of generations", type=int)
parser.add_argument("--parentnum", default = 50, help = "Set the number of parents", type=int)
parser.add_argument("--mu", default = .03, help = "Set mutation rate", type=float)

args = parser.parse_args()

alias = Path(args.imagepath).stem

f = Forger(args.imagepath, population_size=args.popsize, generation_num=args.gennum, parent_num=args.parentnum, dynamic_path=args.dpath, dynamic_mutation=args.dmu)

if args.p:
    results = f.probabilistic_fabricate(mu = args.mu, foldername = alias)
else:
    results = f.fabricate(mu = args.mu, foldername = alias)

plt.plot(results)
plt.title(f'{alias}')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.savefig(PurePath(alias, 'fitness.png'), dpi=600)
plt.show()