# TransfIGN
TransfIGN is a structure-based DL method based on InteractionGraphNet and transformer to predict the interactions between HLA-A 02:01 and antigen peptides.

# How to use
## 1. Data prepare
Unlike sequence-based method, TransfIGN makes predictions based on the 3D structure of peptide-MHC complex, hence we need to generate 3D structure complex by virtual mutation.
We used SCWRL4 for virtual mutation and Amber20 for energy minimization.

### 1. Virtual mutation
python3 ./virtual_mutation/1AKJ.py                                                      #use 1AKJ as template to generate 3D structure for sequence in ./virtual_mutation/sequence.txt We used a dataset with 64 sequences as an example.

### 2. Energy minimization
module load amber/20 && python3 ./virtual_mutation/EM.py                                #get PDB files in ./aesult

### 3. Separate peptide and MHC using chimera
module load chimera && python3 ./virtual_mutation/separate.py --complexpath ./aesult    #generate protein and ligand files

### 4. Generate TransfIGN input files
module load chimera && python3 ./select_residues_pdb.py --pdbpath ./protein --sdfpath ./ligand --finalpath ./ign_input  #Congratulation! You get the TransfIGN input files for DL model training.
