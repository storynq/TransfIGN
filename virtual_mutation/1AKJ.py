import os

def pdb_replace(pdbfile, sample_text):
    before = pdbfile[2]

    after = before.replace('ILKEPVHGV', sample_text )
    newfile = pdbfile[: 2] + [after] + pdbfile[3 :]
    new_string = ''.join(newfile)

    temp_file = open('temp.pdb', 'w')
    temp_file.write(new_string)
    temp_file.close()
    cmd = '/home/nanqi/scwrl4/Scwrl4 -i 1AKJ.pdb -o result_{0}.pdb -s temp.pdb'.format(sample_text)
    ret = os.system(cmd)
    if ret:
        print('cmd error', sample_text)




samplepath = 'sequence.txt'
sampletoken = open(samplepath,'r').readlines()

pdbpath = 'template_1AKJ.txt'
pdbtoken = open(pdbpath, 'r')
pdbfile = pdbtoken.readlines()



for i in sampletoken:
    i = i.rstrip()
    print('current: ', i)
    if len(i) == 9:
        pdb_replace(pdbfile, i)
    else:
        print('sample error', i)
        break
