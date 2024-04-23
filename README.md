# TransfIGN
TransfIGN is a structure-based DL method based on InteractionGraphNet and transformer to predict the interactions between HLA-A 02:01 and antigen peptides.

# Dataset
Our training and validation dataset was compiled from two datasets: Binding affinity (BA) dataset consisting 9051 data points sourced from OnionMHC, eluted ligand (EL) dataset was derived from TransPHLA. This dataset is provided in ./data/train.
Our test set was assembled from the IEDB weekly benchmark dataset starting from 2014 and can be categorized into three subsets: IC50, Binary and T1/2. Test set is provided in ./data/test.

# How to use
## 1. Data prepare
Unlike sequence-based method, TransfIGN makes predictions based on the 3D structure of peptide-MHC complex, hence we need to generate 3D structure complex by virtual mutation.
We used SCWRL4 for virtual mutation, Amber20 for energy minimization and chimera for input file generation.
### 1. Virtual mutation
cd virtual_mutation
python3 1AKJ.py                                                      #use 1AKJ as template to generate 3D structure for sequence in sequence.txt We randomly selected 30 sequences in training set as an example, and this may cost several minutes.
### 2. Energy minimization
module load amber/20 && python3 EM.py                                #get PDB files in ./aesult
### 3. Separate peptide and MHC using chimera
module load chimera && python3 separate.py --complexpath ./aesult    #generate protein and ligand files
### 4. Generate TransfIGN input files
module load chimera && python3 select_residues_pdb.py --pdbpath ./protein --sdfpath ./ligand --finalpath ./ign_input  #Congratulation! You get the TransfIGN input files for DL model training or prediction. We also provide these ign_input files in ign_input_examples for further test.

## 2. Model Training
python3 ./codes_transfIGN/train.py         # Graphs used for GNN are saved in ./codes_transfIGN/train_graph and ./codes_transfIGN/valid_graph. 

## 3. Prediction
python3 ./codes_transfIGN/prediction.py --model_path ./model_save/example.pth --input_path ./prediction_example  #we provide an example model in model_save and some data from Binary test set in prediction_example for prediction.
