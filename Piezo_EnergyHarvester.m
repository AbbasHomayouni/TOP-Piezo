% A 2D TOPOLOGY OPTIMIZATION CODE FOR PIEZOELECTRIC ENERGY HARVESTER
function Piezo_EnergyHarvester
%% GENERAL DEFINITIONS
Lp = 3e-2; % Pieozoelectric plate length (m) in x direction
Wp = 1e-2; % Pieozoelectric plate width (m) in y direction
h = 1e-4; % Pieozoelectric plate Thickness (m) in z direction
nelx = 180; % Number of element in x direction
nely = 60; % Number of element in y direction
penalKuu = 3; penalKup = 6;penalKpp = 4;penalPol = 1; % Penalization factors
omega = 0; % Excitation frequency (Hz)
wj = 1; % Objective function weigthing factor
volfrac = 0.4; % Volume fraction
rmin = 3; % Filter radius
ft = 2; % Filter type - 1 for sensitivity, 2 for density
Max_loop = 400; % Maximum number of Iteration
%% MATERIAL PROPERTIES (PZT 4)
ro = 7500; % Density of piezoelectric material
e31 = -14.9091; % e31 Coupling coefficient
ep33 = 7.8374e-09; % Piezoelectric permitivity epsilon33
C = zeros(3,3); % Creation of null mechanical stiffness tensor
C(1,1) = 9.1187e+10; C(2,2) = C(1,1);
C(1,2) = 3.0025e+10; C(2,1) = C(1,2);
C(3,3) = 3.0581e+10;
%% PREPARE FINITE ELEMENT ANALYSIS
le = Lp/nelx; % Element length
we = Wp/nely; % Element width
e = [e31,e31,0]; % Piezoelectric matrix
x1 = 0;y1 = 0;x2 = le;y2 = 0;x3 = le;y3 = we;x4 = 0;y4 = we; % Element node coordinate
GP = [-1/sqrt(3) -1/sqrt(3);1/sqrt(3) -1/sqrt(3);1/sqrt(3) 1/sqrt(3);-1/sqrt(3) 1/sqrt(3)]; % Gauss quadrature points
kuu = 0;kpp = 0;kup = 0;m = 0; % Initial values for piezoelectric matrices
for i = 1:4
    s = GP(i,1);t = GP(i,2); % Natural coordinates
    n1 = (1/4)*(1-s)*(1-t);
    n2 = (1/4)*(1+s)*(1-t);
    n3 = (1/4)*(1+s)*(1+t);
    n4 = (1/4)*(1-s)*(1+t);
    a = (y1*(s-1)+y2*(-1-s)+y3*(1+s)+y4*(1-s))/4;
    b = (y1*(t-1)+y2*(1-t)+y3*(1+t)+y4*(-1-t))/4;
    c = (x1*(t-1)+x2*(1-t)+x3*(1+t)+x4*(-1-t))/4;
    d = (x1*(s-1)+x2*(-1-s)+x3*(1+s)+x4*(1-s))/4;
    B1 = [a*(t-1)/4-b*(s-1)/4 0 ; 0 c*(s-1)/4-d*(t-1)/4 ;c*(s-1)/4-d*(t-1)/4 a*(t-1)/4-b*(s-1)/4];
    B2 = [a*(1-t)/4-b*(-1-s)/4 0 ; 0 c*(-1-s)/4-d*(1-t)/4;c*(-1-s)/4-d*(1-t)/4 a*(1-t)/4-b*(-1-s)/4];
    B3 = [a*(t+1)/4-b*(s+1)/4 0 ; 0 c*(s+1)/4-d*(t+1)/4 ;c*(s+1)/4-d*(t+1)/4 a*(t+1)/4-b*(s+1)/4];
    B4 = [a*(-1-t)/4-b*(1-s)/4 0 ; 0 c*(1-s)/4-d*(-1-t)/4 ;c*(1-s)/4-d*(-1-t)/4 a*(-1-t)/4-b*(1-s)/4];
    Bfirst = [B1 B2 B3 B4];
    Jfirst = [0 1-t t-s s-1 ; t-1 0 s+1 -s-t ;s-t -s-1 0 t+1 ; 1-s s+t -t-1 0];
    J = [x1 x2 x3 x4]*Jfirst*[y1 ; y2 ; y3 ; y4]/8; % Determinant of jacobian matrix
    Bu = Bfirst/J;
    Bphi = 1/h;
    kuu = kuu + h*J*transpose(Bu)*C*Bu; % Mechanical stiffness matrix
    kup = kup + h*J*transpose(Bu)*e'*Bphi; % Piezoelectric coupling matrix
    kpp = kpp + h*J*transpose(Bphi)*ep33*Bphi; % Dielectric stiffness matrix
    N = [n1,0,n2,0,n3,0,n4,0;0,n1,0,n2,0,n3,0,n4]; % Matrix of interpolation functions
    m = m+J*ro*h*(N')*N; % Mass matrix
end
k0 = max(abs(kuu(:)));beta = max(kpp(:));alpha = max(kup(:));M0 = max(m(:)); % Normalization Factors
kuu = kuu/k0;kup = kup/alpha;kpp = kpp/beta;gamma = (k0*beta)/(alpha^2);m = m/M0; omega = M0*(omega*2*pi)^2/k0; % Normalization
ndof = 2*(nely+1)*(nelx+1); % mechanical degrees of freedom
nele = nelx*nely; % number of elements
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nele,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nele,1);
edofMatPZT = 1:nele;
iK = kron(edofMat,ones(8,1))';
jK = kron(edofMat,ones(1,8))';
iKup = edofMat';
jKup = kron(edofMatPZT,ones(1,8))';
B = ones(nele,1); % Boolean Matrix defined as a vector of ones
%% DEFINITION OF BOUNDARY CONDITION
fixeddofs = 1:2*(nely+1); % Clamped-Free
freedofs  = setdiff(1:ndof,fixeddofs);
lf = length(freedofs);
%% FORCE DEFINITION
nf = 1; % Number of forces
F = sparse(ndof,nf);
Fe = ndof-(nely); % Definition of desired Dof for application of force
F(Fe,1) = +1; % Amplitude of the force
Ftot = [F(freedofs,:);zeros(1,nf)];
%% PREPARE FILTER
iH = ones(nele*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH));
k = 0;
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely+j1;
        for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
            for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                e2 = (i2-1)*nely+j2;
                k = k+1;
                iH(k) = e1;
                jH(k) = e2;
                sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2));
            end
        end
    end
end
H = sparse(iH,jH,sH);
Hs = sum(H,2);
%% INITIALIZE ITERATION
x = repmat(volfrac,nely,nelx); % Initial values for density ratios
pol = repmat(0.1,[nely,nelx]); % Initial values for polarization
xPhys = x;
loop = 0;
Density_change = 1;
E0 = 1; Emin = 1e-9;
e0 = 1; eMin = 1e-9;
eps0 = 1; epsMin = 1e-9;
%% MMA Preparation
mc = 1; % Number of constraints
nVar = 2*nele; % Number of variables
xmin = (1e-9)*ones(nele,1); % Minimum possible density
polmin = 0*ones(nele,1); % Minimum possible polarization
xmin = [xmin;polmin]; % Vector of minimum optimization variables
xmax = ones(nVar,1); % Vector of maximum optimization variables
xold1 = [x(:);pol(:)]; % Vector of variables for previous iteration
xold2 = [x(:);pol(:)]; % Vector of variables for 2nd previous iteration
low = xmin; % Initial vector of lower asymptotes
upp = xmax; % Initial vector of upper asymptotes
a0 = 1;
ai = zeros(mc,1);
ci = (1e5)*ones(mc,1);
di = zeros(mc,1);
%% START ITERATION
while Density_change > 0.005 && loop < Max_loop
    tic
    loop = loop + 1;
    % FE-ANALYSIS
    sM = m(:)*xPhys(:)';
    sKuu = kuu(:)*(Emin+xPhys(:)'.^penalKuu*(E0-Emin));
    sKup = kup(:)*(eMin+xPhys(:)'.^penalKup*(e0-eMin).*((2*pol(:)-1)'.^penalPol));
    sKpp = kpp(:)*(epsMin+xPhys(:)'.^penalKpp*(eps0-epsMin));
    % Creation of matrices
    M = sparse(iK(:),jK(:),sM(:)); % Global masss matrix
    Kuu = sparse(iK,jK,sKuu)-omega*M;
    Kup = sparse(iKup(:),jKup(:),sKup(:)); % Global piezoelectric coupling matrix
    Kpp = sparse(edofMatPZT(:),edofMatPZT(:),sKpp(:)); % Global dielectric stifness matrix
    KupEqui = Kup(freedofs,:)*B; KppEqui = B'*Kpp*B;     % Equipotential Condition
    Ktot = [Kuu(freedofs,freedofs),KupEqui;KupEqui',-gamma*KppEqui]; % Creation of total matrix with equipotential hypothesis
    Ktot = 1/2*(Ktot + Ktot'); % Numerical symmetry enforcement
    U = Ktot\Ftot; % Response vector
    Uu(freedofs,:) = U(1:lf,:); Up = U(lf+1:end,:); % Separation of mechanical displacement and electrical Potential
    ADJ1 = Ktot\[-Kuu(freedofs,freedofs)*Uu(freedofs,:);zeros(1,nf)]; % First adjoint vector
    lambda1(freedofs,:) = ADJ1(1:lf,:); mu1 = B*ADJ1(lf+1:end,:);
    ADJ2 = Ktot\[zeros(lf,nf);-KppEqui*Up]; % Second adjoint vector
    lambda2(freedofs,:) = ADJ2(1:lf,:); mu2 = B*ADJ2(lf+1:end,:);
    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    c = 0; Wm = 0; We = 0;
    dc = zeros(nely,nelx);
    dp = zeros(nely,nelx);
    for i = 1:nf  % nf is the total number of forces
        Uu_i = Uu(:,i);Up_i = B*Up(:,i);
        lambda1_i = lambda1(:,i); lambda2_i = lambda2(:,i);
        mu1_i = mu1(:,i);mu2_i = mu2(:,i);
        Wm = Wm+ reshape(sum((Uu_i(edofMat)*kuu).*Uu_i(edofMat),2),nely,nelx);
        We = We+ reshape(sum((Up_i*kpp).*Up_i,2),nely,nelx);
        dcKuuE = wj*((((1/2)*Uu_i(edofMat) + lambda1_i(edofMat))*kuu).*Uu_i(edofMat))-(1-wj)*((lambda2_i(edofMat)*kuu).*Uu_i(edofMat));
        dcKupE = wj*((lambda1_i(edofMat)*kup).*Up_i + ((Uu_i(edofMat))*kup).*mu1_i)-(1-wj)*((lambda2_i(edofMat)*kup).*Up_i + ((Uu_i(edofMat))*kup).*mu2_i);
        dcKppE = wj*((-mu1_i*kpp).*Up_i)-(1-wj)*((1/2)*(Up_i*kpp).*Up_i - (mu2_i*kpp).*Up_i);
        dcME = wj*((((1/2)*Uu_i(edofMat) + lambda1_i(edofMat))*(-m*omega)).*Uu_i(edofMat))-(1-wj)*((lambda2_i(edofMat)*(-m*omega)).*Uu_i(edofMat));
        dcKuu = reshape(sum(dcKuuE,2),[nely,nelx]);
        dcKup = reshape(sum(dcKupE,2),[nely,nelx]);
        dcKpp = gamma*reshape(sum(dcKppE,2),[nely,nelx]);
        dcM = reshape(sum(dcME,2),[nely,nelx]);
        dc = dc + penalKuu*(E0-Emin)*xPhys.^(penalKuu-1).*dcKuu+penalKup*(e0-eMin)*xPhys.^(penalKup-1).*dcKup.*((2*pol-1).^(penalPol))+penalKpp*(eps0-epsMin)*xPhys.^(penalKpp-1).*dcKpp+dcM;
        dp = dp + (e0-eMin)*2*penalPol*((2*pol-1).^(penalPol-1)).*xPhys.^penalKup.*dcKup;% Polarization sensitivity
    end
    Wm = sum(sum((Emin+xPhys.^penalKuu*(E0-Emin)).*Wm)); % Mechanical energy
    We = sum(sum((epsMin+xPhys.^penalKpp*(eps0-epsMin)).*We)); % Electrical energy
    c = wj*Wm-(1-wj)*We; % Objective function
    dv = ones(nely,nelx);
    % FILTERING/MODIFICATION OF SENSITIVITIES
    if ft == 1
        dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
    elseif ft == 2
        dc(:) = H*(dc(:)./Hs);
        dv(:) = H*(dv(:)./Hs);
    end
    %% MMA OPTIMIZATION OF DESIGN VARIABLES
    dp = dp/max(abs(dp(:))); % Normalizing the polarization sensitivity
    xval = [x(:);pol(:)]; % Vector of current optimization variables
    f0val = c; % Current objective function value
    df0dx = [dc(:);dp(:)]; % Vector of Sensitivities
    fval = [sum(xPhys(:))/(volfrac*nele) - 1]; % Constraint value
    dfdx = [dv(:)' / (volfrac*nele),0*pol(:)']; % Constraint's Sensitivities
    [xmma, ~, ~, ~, ~, ~, ~, ~, ~, low,upp] = mmasub(mc, nVar, loop, xval, xmin, xmax, xold1, xold2, ...
        f0val,df0dx,fval,dfdx,low,upp,a0,ai,ci,di); % MMA optimization
    xnew = reshape(xmma(1:nele,1),nely,nelx); % Vector of updated density variables
    if ft == 1
        xPhys = xnew;
    elseif ft == 2
        xPhys(:) = (H*xnew(:))./Hs;
    end
    Density_change = max(abs(xnew(:)-x(:)));
    xold2 = xold1(:);
    xold1 = [x(:);pol(:)];
    pol = reshape(xmma(nele+1:2*nele,1),nely,nelx); % Vector of updated polarization variables
    x = xnew;
    %% PLOT DENSITIES & POLARIZATION
    figure(1);colormap(gray); imagesc(1-x); caxis([0 1]); axis equal; axis off; drawnow;
    figure(2);colormap(jet); imagesc(((x.*(pol*2-1))+1)/2);  caxis([0 1]); axis equal; axis off; drawnow;
    fprintf(' It:%2.0i Time:%3.2fs Obj:%3.4f Wm.:%3.4f We.:%3.4f Vol:%3.3f ch:%3.3f\n ',loop,toc,c,Wm,We,mean(xPhys(:)),Density_change);
end
% ||=====================================================================||
% || THIS CODE IS WRITTEN BY ABBAS HOMAYOUNI-AMLASHI, THOMAS SCHLINQUER, ||
% ||    ABDENBI MOHAND-OUSAID AND MICKY RAKOTONDRABE                     ||
% ||=====================================================================||