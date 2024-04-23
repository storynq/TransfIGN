#!/bin/bash
pdb4amber -i sample.pdb -o output.pdb -y --dry
reduce output.pdb > sample_H.pdb
pdb4amber -i sample_H.pdb -o sample_new.pdb
tleap -f tleap.in
$AMBERHOME/bin/pmemd.cuda -O -i md.in -o 01_Min.out -p x.prmtop -c x.inpcrd -r 01_Min.ncrst -inf 01_Min.mdinfo
cpptraj -p x.prmtop -y 01_Min.ncrst -x y.rst
ambpdb -p x.prmtop <y.rst> result1.pdb
mv result1.pdb aesult