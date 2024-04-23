# TransfIGN
TransfIGN is a structure-based DL method based on InteractionGraphNet and transformer to predict the interactions between HLA-A 02:01 and antigen peptides.

# Dataset
Our training and validation dataset was compiled from two datasets: Binding affinity (BA) dataset consisting of 9051 data points sourced from OnionMHC, eluted ligand (EL) dataset was derived from TransPHLA. This dataset is provided in ./data/train.  

Our test set was assembled from the IEDB weekly benchmark dataset starting from 2014 and can be categorized into three subsets: IC50, Binary and T1/2. Test set is provided in ./data/test.  

For binding affinity prediction comparison, the IC50 values are provided in ./data/test/IC50_RMSE.xlsx

# Environment Perparation
This program is using DGL and Pytorch.
Environment package file can be download [here](https://drive.google.com/file/d/1Rls2ydUSoEjW_rRnvXBzBCcoB4YvcWLQ/view).   
After unpacking the file, environment can be activated by:  
```
source /unpack_file/bin/activate
```

# Usage
## 1. Data prepare
Unlike sequence-based method, TransfIGN makes predictions based on the 3D structure of peptide-MHC complex, hence we need to generate 3D structure complex by virtual mutation.
We used SCWRL4 for virtual mutation, Amber20 for energy minimization and chimera for input file generation.

### 1. Virtual mutation  
Use 1AKJ as template to generate 3D structure for sequences in ./virtual_mutation/sequence.txt We randomly selected 30 sequences in training set as an example, and this may cost several minutes.  

```
cd virtual_mutation  
python3 1AKJ.py
```
                                     
### 2. Energy minimization
Save PDB files in ./aesult  
```
module load amber/20 && python3 EM.py                                
```

### 3. Separate peptide and MHC using chimera
Generate protein and ligand files  
```
module load chimera && python3 separate.py --complexpath ./aesult    
```

### 4. Generate TransfIGN input files
```
module load chimera && python3 select_residues_pdb.py --pdbpath ./protein --sdfpath ./ligand --finalpath ./ign_input
```

Congratulation! You get the TransfIGN input files for DL model training or prediction. We also provide these ign_input files in ./virtual_mutation/ign_input_examples for further test.

## 2. Model Training
Graphs used for GNN are saved in ./codes_transfIGN/train_graph and ./codes_transfIGN/valid_graph. The training results will be save in ./stats.  
```
python3 ./codes_transfIGN/train.py         
```

## 3. Prediction
We provide an example model in model_save and some data from Binary test set in prediction_example for prediction. The prediction results will be saved in ./stats  
```
python3 ./codes_transfIGN/prediction.py --model_path ./model_save/example.pth --input_path ./prediction_example  
```


