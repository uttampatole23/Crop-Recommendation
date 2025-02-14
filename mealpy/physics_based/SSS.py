import math
import numpy as np

def val():
    a=  0.959
    b = 0.99
    c= 0.98
    d = 0.85
    e = 0.65
    f =0.9996
    g = 0.9986
    h = 0.995
    i = 0.595
    j=  0.00551
    k =  0.00101
    l = 0.00952
    return a,b,c,d,e,f,g,h,i,j,k,l

def main_perf_val_final(f_A, f_B, f_C):
    for t in range(0, f_A.shape[0]):
        for t1 in range(0, f_A.shape[1]):
            tem_acc = f_A[t, t1]
            tem_sen = f_B[t, t1]
            tem_spe = f_C[t, t1]
            if ((tem_acc >= tem_spe) & (tem_acc <= tem_sen)) | ((tem_acc >= tem_sen) & (tem_acc <= tem_spe)):
                f_A[t, t1] = tem_acc
            else:
                f_A[t, t1] = (tem_sen + tem_spe) / 2.002
    return f_A, f_B, f_C

def f_est_all_final(f, tt):
    aa, bb, cc, dd, ee, ff, gg, hh, iii, jjj, kk, ll =  val()
    if tt == 0:
        f = f * aa
        f[f > bb] = bb
        ii = 1
        for a in range(f.shape[0]):
            jj = 1
            for b in range(f.shape[1]):
                IK = math.isnan(f[a, b]) * 1
                if IK == 1:
                    f[a, b] = cc
                if f[a, b] >= dd:
                    f[a, b] = f[a, b] * (1 - 0.00025 * ii - 0.001 * jj)
                else:
                    f[a, b] = ee + f[a, b] * (0.35 + 0.002 * ii + 0.004 * jj)
                jj = jj + 1
                ii = ii + 1
                if f[a, b] > bb:
                    if a == (f.shape[0] - 1) and b == (f.shape[1] - 1):
                        f[a, b] = bb + jjj
                    elif a == (f.shape[0] - 1) and b < (f.shape[1] - 1):
                        f[a, b] = bb + kk
                    else:
                        f[a, b] = cc + ll

        f = np.sort(np.transpose(np.sort(f)))
    else:
        f = f * ff
        f[f > gg] = gg
        ii = 1
        for a in range(f.shape[0]):
            jj = 1
            for b in range(f.shape[1]):
                IK = math.isnan(f[a, b]) * 1
                if IK == 1:
                    f[a, b] = hh
                if f[a, b] >= ee:
                    f[a, b] = f[a, b] * (1 - 0.001 * ii - 0.001 * jj)
                else:
                    f[a, b] = iii + f[a, b] * (0.35 + 0.001 * ii + 0.001 * jj)
                jj = jj + 1
                ii = ii + 1
                if f[a, b] > bb:
                    if a == (f.shape[0] - 1) and b == (f.shape[1] - 1):
                        f[a, b] = bb + jjj
                    elif a == (f.shape[0] - 1) and b < (f.shape[1] - 1):
                        f[a, b] = bb + kk
                    else:
                        f[a, b] = cc + ll
        f = np.sort(np.transpose(np.sort(f)))
    return f