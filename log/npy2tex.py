import numpy as np 

arr = np.load('./log/predEvalArray.npy')
arr = arr.T
l = []
l.append(arr[1]/arr[0]) #acc
l.append(arr[2]/arr[3]) #rec.
l.append(arr[2]/arr[4]) #prec.
l.append(2*l[1]*l[2]/(l[1]+l[2])) #f1
name = ['acc  ', 'rec. ', 'prec.', 'F1   ']
for i, eval in enumerate(l):
    print('&%s'%name[i], end=' ')
    for e in eval:
        print('&%.2f'%(e*100), end=' ')
    print("&%.2f\\\\"%(eval.sum()/8*100))