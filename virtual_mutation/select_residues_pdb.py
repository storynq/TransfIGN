import os
from rdkit import Chem
import pickle
import multiprocessing
from itertools import repeat
import argparse
path_marker = '/'

# adding deleting temporary files(pocket file and py file)
def rename_files(path):
    old_names = os.listdir(path)
    for old_name in old_names:
        if old_name.endswith('ligand'):
            new_name1 = old_name.replace('aesult_','')
            new_name2 = new_name1.replace('_ligand','')
            os.rename(os.path.join(path,old_name), os.path.join(path,new_name2))

def generate_complex(protein_file, ligand_file):
    # get the pocket of protein
    # protein = Chem.MolFromPDBFile(receptor)
    # the content of python file for Chimera
    pocket_file = pocketpath + path_marker + ligand_file.split('/')[-1].replace('.pdb', '_pocket.pdb')
    filecontent = "from chimera import runCommand \n"
    filecontent += "runCommand('open 0 %s') \n" % ligand_file
    filecontent += "runCommand('open 1 %s') \n" % protein_file
    filecontent += "runCommand('select #1 & #0 z < 10') \n"
    filecontent += "runCommand('write format pdb selected 1 %s') \n" % pocket_file
    filecontent += "runCommand('close 0') \n"
    filecontent += "runCommand('close 1')"
    filename = pypath + path_marker + ligand_file.split('/')[-1].replace('.pdb', '.py')
    with open(filename, 'w') as f:
        f.write(filecontent)

    try:
        cmdline = 'chimera --nogui --silent --script %s' % filename
        os.system(cmdline)

        # ligand = Chem.SDMolSupplier(ligand_file)[0]
        ligand = Chem.MolFromPDBFile(ligand_file)
        pocket = Chem.MolFromPDBFile(pocket_file)
        # write the ligand and pocket to pickle object
        # ComplexFileName = ''.join(['./ign_input/', ligand_file.split('/')[-1].strip('.sdf')])
        ComplexFileName = ''.join([finalpath+path_marker, ligand_file.split('/')[-1][:-4]])
        with open(ComplexFileName, 'wb') as ComplexFile:
            pickle.dump([ligand, pocket], ComplexFile)
    except:
        print('complex %s generation failed...' % ligand_file)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--pdbpath', type=str, default='./examples/protein',
                           help="the relative path for protein files")
    argparser.add_argument('--sdfpath', type=str, default='./examples/ligand',
                           help="the relative path for storing converted sdf files")
    argparser.add_argument('--pocketpath', type=str, default='./examples/chimera_pocket',
                           help="the relative path for storing pocket files (temporary files)")
    argparser.add_argument('--pypath', type=str, default='./examples/chimera_py',
                           help="the relative path for storing .py files for running chimera(temporary files)")
    argparser.add_argument('--finalpath', type=str, default='./examples/ign_input',
                           help="the relative path for storing files for ign input")
    argparser.add_argument('--num_process', type=int, default=12,
                           help="the number of process for generating ign inputs")
    args = argparser.parse_args()
    pdbpath, sdfpath, pocketpath, pypath, finalpath = args.pdbpath, args.sdfpath, args.pocketpath, args.pypath, args.finalpath
    num_process = args.num_process
    if not os.path.exists(pocketpath):
        os.makedirs(pocketpath)
    if not os.path.exists(pypath):
        os.makedirs(pypath)
    if not os.path.exists(finalpath):
        os.makedirs(finalpath)

    ligands = os.listdir(sdfpath)
    proteins = os.listdir(pdbpath)
    ligands.sort()
    proteins.sort()
    ligandfiles = [sdfpath + path_marker+ligand for ligand in ligands]
    proteinfiles = [pdbpath + path_marker + protein for protein in proteins]

    pool = multiprocessing.Pool(num_process)
    pool.starmap(generate_complex, zip(proteinfiles, ligandfiles))
    pool.close()
    pool.join()

    rename_files(finalpath)

    # remove the temporary files
    # cmdline = 'rm -rf %s &&' % (pocketpath + path_marker + '*')
    # cmdline += 'rm -rf %s' % (pypath + path_marker + '*')
    # os.system(cmdline)
