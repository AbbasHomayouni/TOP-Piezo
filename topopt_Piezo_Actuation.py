# 2D Pizo_Actuator Topology Optimization Code written by Abbas Homayouni-Amlashi
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
import math
import time
# General Definitions
# geometrical dimension of the piezoelectric plate
Lp, Wp, h = [1e-2, 0.5e-2, 1e-4]
nelx, nely = [150, 75]  # resolution of the mesh
penalKuu, penalKup, penalPol = [3, 4, 1]  # penalization factors
Ks = .005
volfrac = 0.3
rmin = 2.5
ft = 0
Max_loop = 250
move = 0.2
# Material Properties
e31 = -14.9091
C = np.zeros((3, 3), dtype=float)
C[0, 0], C[1, 1], C[0, 1], C[1, 0], C[2, 2] = [
    9.1187e+10, 9.1187e+10, 3.0025e+10, 3.0025e+10, 3.0581e+10]

# PREPARE FINITE ELEMENT ANALYSIS
le, we = [Lp/nelx, Wp/nely]  # element size
e = [e31, e31, 0]  # Piezoelectric matrix
x1, y1, x2, y2, x3, y3, x4, y4 = [0, 0, le, 0,
                                  le, we, 0, we]  # Element node coordinates
GP = np.zeros((4, 2), dtype=float)  # nul matrix for definition of gauss points
GP_D = 1/(math.sqrt(3))  # gauss value for 2 point gauss quadrature method
GP[0, 0], GP[0, 1], GP[1, 0], GP[1, 1], GP[2, 0], GP[2, 1], GP[3, 0], GP[3,
                                                                         1] = [-GP_D, -GP_D, GP_D, -GP_D, GP_D, GP_D, -GP_D, GP_D]  # definition of gauss matrix
kuu, kup = [0, 0]  # Initial values for piezoelectric matrices
B = np.zeros((3, 8), dtype=float)
for i in range(4):
    [s, t] = [GP[i, 0], GP[i, 1]]
    n1, n2, n3, n4 = [0.25*(1-s)*(1-t), 0.25*(1+s)*(1-t),
                      0.25*(1+s)*(1+t), 0.25*(1-s)*(1+t)]
    a = (y1*(s-1)+y2*(-1-s)+y3*(1+s)+y4*(1-s))/4
    b = (y1*(t-1)+y2*(1-t)+y3*(1+t)+y4*(-1-t))/4
    c = (x1*(t-1)+x2*(1-t)+x3*(1+t)+x4*(-1-t))/4
    d = (x1*(s-1)+x2*(-1-s)+x3*(1+s)+x4*(1-s))/4
    B[0, 0], B[0, 2], B[0, 4], B[0, 6] = [
        a*(t-1)/4-b*(s-1)/4, a*(1-t)/4-b*(-1-s)/4, a*(t+1)/4-b*(s+1)/4, a*(-1-t)/4-b*(1-s)/4]
    B[1, 1], B[1, 3], B[1, 5], B[1, 7] = [
        c*(s-1)/4-d*(t-1)/4, c*(-1-s)/4-d*(1-t)/4, c*(s+1)/4-d*(t+1)/4, c*(1-s)/4-d*(-1-t)/4]
    B[2, 0], B[2, 2], B[2, 4], B[2, 6] = [
        c*(s-1)/4-d*(t-1)/4, c*(-1-s)/4-d*(1-t)/4, c*(s+1)/4-d*(t+1)/4, c*(1-s)/4-d*(-1-t)/4]
    B[2, 1], B[2, 3], B[2, 5], B[2, 7] = [
        a*(t-1)/4-b*(s-1)/4, a*(1-t)/4-b*(-1-s)/4, a*(t+1)/4-b*(s+1)/4, a*(-1-t)/4-b*(1-s)/4]
    Jfirst = np.array([[0, 1-t, t-s, s-1], [t-1, 0, s+1, -s-t,],
                      [s-t, -s-1, 0, t+1], [1-s, s+t, -t-1, 0]])
    J = np.array([x1, x2, x3, x4]).dot(Jfirst.dot(
        np.array([[y1/8], [y2/8], [y3/8], [y4/8]])))  # Determinant of jacobian matrix
    Bu = B/J
    Bphi = 1/h
    kuu = kuu+h*J*(np.transpose(Bu).dot(C)).dot(Bu)
    kup = kup + h*J*(np.transpose(Bu).dot(np.transpose(e).dot(Bphi)))
abs_kuu, abs_kup = [np.absolute(kuu), np.absolute(kup)]
k0, alpha = [abs_kuu.max(), abs_kup.max()]  # Normalization Factors
kuu, kup = [kuu/k0, kup/alpha]  # Normalization
ndof = 2*(nely+1)*(nelx+1)  # Mechanical degrees of freedom
nele = nelx*nely  # Number of elements
edofMat = np.zeros((nelx*nely, 8), dtype=int)
for elx in range(nelx):
    for ely in range(nely):
        el = ely+elx*nely
        n1 = (nely+1)*elx+ely
        n2 = (nely+1)*(elx+1)+ely
        edofMat[el, :] = np.array(
            [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])

edofMatPZT = list(range(nele))
# Construct the index pointers for the coo format
iK = np.kron(edofMat, np.ones((8, 1))).T
jK = np.kron(edofMat, np.ones((1, 8))).T
iKup = np.transpose(edofMat)
jKup = np.transpose(np.kron(edofMatPZT, np.ones((1, 8))))
# OUTPUT DISPLACEMENT DEFINITION
DMDOF = ndof-2  # Desired mechanical degree of freedom
lenDMDOF = 1
L = np.zeros((2*(nely+1)*(nelx+1), 1), dtype=int)
L[DMDOF, 0] = -1
Uu = np.zeros((ndof, 1), dtype=float)  # Creation of null displacement vector
Adjoint = np.zeros((ndof, 1), dtype=float)  # Creation of null adjoint vector
Up = np.ones((nele, 1), dtype=float)  # Actuation voltage
# DEFINITION OF BOUNDARY CONDITION
fixeddofs1 = list(range(2*(nely+1)))  # Main supports
fixeddofs2 = np.arange((nely+1), ((nely+1)*(nelx+1))+1,
                       (nely+1)).dot(2)-1  # Applying symmetry
fixeddofs = np.union1d(fixeddofs1, fixeddofs2)  # Fusion of every supports
# Computation of freedofs
freedofs = np.setdiff1d(list(range(ndof)), fixeddofs)
lf = len(freedofs)  # Number of free dofs
# Filter: Build (and assemble) the index+data vectors for the coo matrix format
nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
iH = np.zeros(nfilter)
jH = np.zeros(nfilter)
sH = np.zeros(nfilter)
cc = 0
for i in range(nelx):
    for j in range(nely):
        row = i*nely+j
        kk1 = int(np.maximum(i-(np.ceil(rmin)-1), 0))
        kk2 = int(np.minimum(i+np.ceil(rmin), nelx))
        ll1 = int(np.maximum(j-(np.ceil(rmin)-1), 0))
        ll2 = int(np.minimum(j+np.ceil(rmin), nely))
        for k in range(kk1, kk2):
            for l in range(ll1, ll2):
                col = k*nely+l
                fac = rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
                iH[cc] = row
                jH[cc] = col
                sH[cc] = np.maximum(0.0, fac)
                cc = cc+1
H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
Hs = H.sum(1)
# Optimality criterion


def oc(nelx, nely, x, xPhys, volfrac, dc, dv, move):
    l1, l2 = [0, 1e9]
    # reshape to perform vector operations
    xnew = np.zeros(nele)
    while (l2-l1)/(np.maximum(l1+l2, 1e-15)) > 1e-3:
        lmid = 0.5*(l2+l1)
        xnew = np.maximum(0.001, np.maximum(
            x-move, np.minimum(1.0, np.minimum(x+move, x*((np.maximum(1e-30, -dc/dv/lmid))**0.3)))))
        if ft == 0:
            xPhys = xnew
        elif ft == 1:
            xPhys = (np.asarray((H*xnew.reshape(nely*nelx, 1, order='F'))/Hs)
                     ).reshape(nely, nelx, order='F')
        if sum(xPhys.flatten(order='F')) > volfrac*nele:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, xPhys)


# INITIALIZE ITERATION
x = volfrac * np.ones((nely, nelx), dtype=float)  # Density initial values
# Initial values for polarization
pol = 0.1 * np.ones((nely, nelx), dtype=float)
xPhys = x.copy()
xold = x.copy()
loop = 0
Density_change = 1
E0, Emin = [1, 1e-9]
e0, eMin = [1, 1e-9]
dc = np.ones(nely*nelx)
dp = np.ones(nely*nelx)
dv = np.ones(nely*nelx)
g = 0  # must be initialized to use the NGuyen/Paulino OC approach
# Initialize plot and plot the initial design
plt.ion()  # Ensure that redrawing is possible
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
im1 = ax1.imshow(np.concatenate((-xPhys, np.flip(-xPhys, 0)), axis=0),
                 cmap='gray', interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
ax1.set_title('Density')
ax1.set_xlabel('nelx')
ax1.set_ylabel('nely')
im2 = ax2.imshow(np.concatenate((-xPhys, np.flip(-xPhys, 0)), axis=0),
                 cmap='jet', interpolation='none', norm=colors.Normalize(vmin=0, vmax=1))
ax2.set_title('Polarization')
ax2.set_xlabel('nelx')
ax2.set_ylabel('nely')
fig.show()
# START ITERATION
while loop < Max_loop:
    timestart = time.time()
    loop = loop + 1
    check = (Emin+(xPhys.reshape(1, nely*nelx, order='F'))**penalKuu*(E0-Emin))
    sKuu = kuu.flatten()[:, np.newaxis]*(Emin+(xPhys.reshape(1,
                                                             nely*nelx, order='F'))**penalKuu*(E0-Emin))
    sKup = kup.flatten()[:, np.newaxis]*(eMin+((xPhys.reshape(1, nely*nelx, order='F'))
                                               ** penalKup)*(e0-eMin))*((2*pol.reshape(1, nely*nelx, order='F')-1)**penalPol)
    Kuu = coo_matrix((sKuu.flatten(order='F'), (iK.flatten(
        order='F'), jK.flatten(order='F'))), shape=(ndof, ndof)).tocsc()
    Kup = coo_matrix((sKup.flatten(order='F'), (iKup.flatten(
        order='F'), jKup.flatten(order='F'))), shape=(ndof, nele)).tocsc()
    # Assembling the stiffness of the modeled spring
    Kuu[DMDOF, DMDOF] = Kuu[DMDOF, DMDOF]+Ks * \
        np.ones((lenDMDOF, lenDMDOF), dtype=float)
    Uu[freedofs, 0] = spsolve(
        Kuu[freedofs, :][:, freedofs], (-Kup[freedofs, :]*Up))
    CE = np.sum(-L*Uu)
    # SENSITIVITY ANALYSIS
    Adjoint[freedofs, 0] = spsolve(
        Kuu[freedofs, :][:, freedofs], L[freedofs, 0])  # Adjoint vector
    DCKuuE = np.sum(np.multiply(np.dot(
        (np.sum(Adjoint[edofMat], axis=2)), kuu), np.sum(Uu[edofMat], axis=2)), axis=1)
    DCKupE = np.multiply(
        np.dot((np.sum(Adjoint[edofMat], axis=2)), kup), Up[edofMatPZT].T)
    DCKuu = (DCKuuE)[:, np.newaxis].reshape(nely, nelx, order='F')
    DCKup = (DCKupE)[:, np.newaxis].reshape(nely, nelx, order='F')
    dc = penalKuu*(E0-Emin)*(xPhys**(penalKuu-1))*DCKuu+penalKup * \
        (E0-Emin)*((2*pol-1)**(penalPol))*(xPhys**(penalKup-1))*DCKup
    dp = (np.multiply(np.multiply(2*penalPol *
          ((2*pol-1)**(penalPol-1)), xPhys**(penalKup)), DCKup))
    dv = np.ones((nely, nelx), dtype=float)
    # Sensitivity filtering
    if ft == 0:
        dc = (np.asarray((H*(x.reshape(nely*nelx, 1, order='F')*dc.reshape(nely*nelx, 1, order='F'))) /
              Hs/np.maximum(1e-3, x.reshape(nely*nelx, 1, order='F')))).reshape(nely, nelx, order='F')
    elif ft == 1:
        dc = (np.asarray(H*(dc.reshape(nely*nelx, 1, order='F')/Hs))
              ).reshape(nely, nelx, order='F')
        dv = (np.asarray(H*(dv.reshape(nely*nelx, 1, order='F')/Hs))
              ).reshape(nely, nelx, order='F')
    # OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES
    (x_new, xPhys) = oc(nelx, nely, x, xPhys, volfrac, dc, dv, move)
    pol = np.maximum(0, np.maximum(
        pol-move, np.minimum(1, np.minimum(pol+move, np.sign(-dp)))))
    Density_change = np.linalg.norm(x_new.reshape(
        nelx*nely, 1) - x.reshape(nelx*nely, 1), np.inf)
    x = x_new
    # PLOT DENSITIES & POLARIZATION
    im1.set_array(np.concatenate((-xPhys, np.flip(-xPhys, 0)), axis=0))
    im2.set_array(np.concatenate((((xPhys*(2*pol-1))+1)/2,
                  ((np.flip(xPhys, 0)*(2*np.flip(pol, 0)-1))+1)/2), axis=0))
    fig.canvas.draw()
    plt.pause(0.01)  # may be try: plt.show()
    timeend = time.time()
    runtime = timeend - timestart
    print("it.: {0} , time.: {1:.3f} , obj.: {2:.3f} Vol.: {3:.3f}, ch.: {4:.3f}".format(
        loop,  runtime, CE, sum(xPhys.flatten(order='F'))/nele, Density_change))
plt.show(block=True)
# print (runtime)
