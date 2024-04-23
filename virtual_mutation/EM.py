import os

samplepath = 'sequence.txt'
sampletoken = open(samplepath,'r').readlines()

testpath = 'test1.sh'
testtoken = open(testpath,'r')
test_text = testtoken.readlines()



def text_replace(test_text,mutant_text):
    pdb_before = test_text[1]
    output_before = test_text[7]
    remove_before = test_text[8]
    pdbname = 'result_{0}.pdb'.format(mutant_text)
    pdb_after = pdb_before.replace('sample.pdb', pdbname)
    resultname = 'aesult_{0}.pdb'.format(mutant_text)
    output_after = output_before.replace('result1.pdb',resultname)
    remove_after = remove_before.replace('result1.pdb',resultname)
    input_text = test_text[0]+ pdb_after+ test_text[2]+ test_text[3]+ test_text[4]+ test_text[5]+ test_text[6]+ output_after+ remove_after
    input_text2 = ''.join(input_text)

    mutant_file = open('input.sh','w')
    mutant_file.write(input_text2)
    mutant_file.close()

    cmd = './input.sh'
    ret = os.system(cmd)
    if ret:
        print('cmd error', mutant_text)


for i in sampletoken:
    i = i.rstrip()
    if len(i) == 9:
        text_replace(test_text,i)
    else:
        print('sample error')
        break
