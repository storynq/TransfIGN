import os
from rdkit import Chem
import pickle
import multiprocessing
from itertools import repeat
import argparse
path_marker = '/'

def division_complex(complex_file,nofile):
    # get the pocket of protein
    # protein = Chem.MolFromPDBFile(receptor)
    # the content of python file for Chimera
    protein_file = proteinpath + path_marker + complex_file.split('/')[-1].replace('.pdb', '_protein.pdb')
    ligand_file = ligandpath + path_marker + complex_file.split('/')[-1].replace('.pdb', '_ligand.pdb')
    filecontent = "from chimera import runCommand \n"
    filecontent += "runCommand('open 0 %s') \n" % nofile
    filecontent += "runCommand('split ligand') \n"
    filecontent += "runCommand('write format pdb 0.2 %s') \n" % ligand_file
    filecontent += "runCommand('write format pdb 0.1 %s') \n" % protein_file
    filename = pypath + path_marker + complex_file.split('/')[-1].replace('.pdb', '.py')
    with open(filename, 'w') as f:
        f.write(filecontent)

    try:
        cmdline = 'chimera --nogui --silent --script %s' % filename
        os.system(cmdline)

    except:
        print('complex %s generation failed...' % ligand_file)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--complexpath', type=str, default='./complex',
                           help="the relative path for complex files")
    argparser.add_argument('--proteinpath', type=str, default='./protein',
                           help="the relative path for generate protein files")
    argparser.add_argument('--ligandpath', type=str, default='./ligand',
                           help="the relative path for generate ligand(pdb) files")
    argparser.add_argument('--pypath', type=str, default='./chimera_py',
                           help="the relative path for storing .py files for running chimera")
    argparser.add_argument('--num_process', type=int, default=12,
                           help="the number of process for generating ign inputs")
    args = argparser.parse_args()
    complexpath, proteinpath, ligandpath, pypath= args.complexpath, args.proteinpath, args.ligandpath, args.pypath,
    num_process = args.num_process
    if not os.path.exists(proteinpath):
        os.makedirs(proteinpath)
    if not os.path.exists(ligandpath):
        os.makedirs(ligandpath)
    if not os.path.exists(pypath):
        os.makedirs(pypath)

    complexs = os.listdir(complexpath)
    complexfiles = [complexpath + path_marker + complex for complex in complexs]
    nofiles = complexfiles

    pool = multiprocessing.Pool(num_process)

    pool.starmap(division_complex, zip(complexfiles,nofiles))
    pool.close()
    pool.join()
    #remove the temporary files
    #cmdline = 'rm -rf %s &&' % (pocketpath + path_marker + '*')
    #cmdline += 'rm -rf %s' % (pypath + path_marker + '*')
    #os.system(cmdline)
