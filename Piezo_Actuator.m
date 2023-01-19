% A TOPOLOGY OPTIMIZATION MATLAB CODE FOR PIEZOELECTRIC ACTUATOR
function Piezo_Actuator
%% GENERAL DEFINITIONS
Lp = 1e-2; % Pieozoelectric plate length (m) in x direction
Wp = 0.5e-2; % Pieozoelectric plate width (m) in y direction
h = 1e-4; % Pieozoelectric plate Thickness (m) in z direction
nelx = 150; % Number of element in x direction
nely = 75; % Number of element in y direction
penalKuu = 3;penalKup = 4;penalPol = 1; % Penalization factors
Ks = 0.005;  %2e-2/3 End spring stiffness
volfrac = 0.3; % Volume fraction
rmin = 2.5; % Filter radius
ft = 1; % Filter type - 1 for sensitivity, 2 for density
Max_loop = 1000; % Maximum number of Iteration
%% MATERIAL PROPERTIES (PZT 4)
e31 = -14.9091; % e31 Coupling coefficient
C = zeros(3,3); % Creation of null mechanical stiffness tensor
C(1,1) = 9.1187e+10;C(2,2) = C(1,1);
C(1,2) = 3.0025e+10;C(2,1) = C(1,2);
C(3,3) = 3.0581e+10;
%% PREPARE FINITE ELEMENT ANALYSIS
le = Lp/nelx; % Element length
we = Wp/nely; % Element width
e = [e31,e31,0]; % Piezoelectric matrix
x1 = 0;y1 = 0;x2 = le;y2 = 0;x3 = le;y3 = we;x4 = 0;y4 = we; % Element node coordinate
GP = [-1/sqrt(3) -1/sqrt(3);1/sqrt(3) -1/sqrt(3);1/sqrt(3) 1/sqrt(3);-1/sqrt(3) 1/sqrt(3)]; % Gauss quadrature DMDOFs
kuu = 0;kup = 0; % Initial values for piezoelectric matrices
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
    kuu = kuu+h*J*transpose(Bu)*C*Bu; % Mechanical stiffness matrix:
    kup = kup+ h*J*transpose(Bu)*e'*Bphi; % Piezoelectric coupling matrix:
end
k0 = max(abs(kuu(:)));alpha = max(kup(:));% Normalization Factors
kuu = kuu/k0;kup = kup/alpha; % Normalization
ndof = 2*(nely+1)*(nelx+1); % Mechanical degrees of freedom
nele = nelx*nely; % Number of elements
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nele,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nele,1);
edofMatPZT = 1:nele;
iK = kron(edofMat,ones(8,1))';
jK = kron(edofMat,ones(1,8))';
iKup = edofMat';
jKup = kron(edofMatPZT,ones(1,8))';
%% OUTPUT DISPLACEMENT DEFINITION
DMDOF = ndof-1; % Desired mechanical degree of freedom
L = sparse(2*(nely+1)*(nelx+1),1);
L(DMDOF,1) = -1;
Uu = zeros(ndof,1); % Creation of null displacement vector
Adjoint = zeros(ndof,1); % Creation of null adjoint vector
Up(1:nele,1)= 1;% Actuation voltage
%% DEFINITION OF BOUNDARY CONDITION
fixeddofs1 = 1:2*(nely+1); % Main supports
fixeddofs2 = 2*[(nely+1):(nely+1):(nely+1)*(nelx+1)]; % Applying symmetry
fixeddofs = [fixeddofs1,fixeddofs2]; % Fusion of every supports
freedofs = setdiff(1:ndof,fixeddofs); % Computation of freedofs
lf = length(freedofs); % Number of free dofs
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
%% START ITERATION
while Density_change > 0.01 && loop < Max_loop
    tic
    loop = loop + 1;
    % FE-ANALYSIS
    sKuu = kuu(:)*(Emin+xPhys(:)'.^penalKuu*(E0-Emin));
    sKup = kup(:)*(eMin+xPhys(:)'.^penalKup*(e0-eMin).*((2*pol(:)-1)'.^penalPol));
    Kuu = sparse(iK,jK,sKuu); % Global stifness matrix
    Kup = sparse(iKup(:),jKup(:),sKup(:)); % Global piezoelectric coupling matrix
    Kuu(DMDOF,DMDOF) = Kuu(DMDOF,DMDOF)+Ks; % Assembling the stifness of the modeled spring
    Uu(freedofs,:) = Kuu(freedofs,freedofs)\(-Kup(freedofs,:)*Up); % Mechanical displacement
    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    CE = -(sum(L(edofMat).*Uu(edofMat),2));
    c = full(sum(CE)); % Objective Function
    Adjoint(freedofs,:) = Kuu(freedofs,freedofs)\L(freedofs,:); % Adjoint vector
    DCKuuE = sum((Adjoint(edofMat)*kuu).*Uu(edofMat),2);
    DCKupE = (Adjoint(edofMat)*kup).*Up(edofMatPZT);
    DCKuu = reshape(DCKuuE,[nely,nelx]);
    DCKup = reshape(DCKupE,[nely,nelx]);
    dc = penalKuu*(E0-Emin)*xPhys.^(penalKuu-1).*DCKuu+...
        penalKup*(E0-Emin)*((2*pol-1).^(penalPol)).*xPhys.^(penalKup-1).*DCKup; % Sensitivity with respect to x
    dp = 2*penalPol*((2*pol-1).^(penalPol-1)).*xPhys.^(penalKup).*DCKup; % Sensitivity with respect to p
    dv = ones(nely,nelx);     % Volume sensitivity
    % FILTERING/MODIFICATION OF SENSITIVITIES
    if ft == 1
        dc(:) = H*(x(:).*dc(:))./Hs./max(1e-3,x(:));
    elseif ft == 2
        dc(:) = H*(dc(:)./Hs); dv(:) = H*(dv(:)./Hs);
    end
    %% OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES
    l1 = 0; l2 = 1e9; move = 0.2;
    while (l2-l1)/(l1+l2) > 1e-3
        lmid = 0.5*(l2+l1);
        xnew = max(0.001,max(x-move,min(1.,min(x+move,x.*(max(1e-30,-dc./dv/lmid)).^0.3)))); % OC update of density
        if ft == 1
            xPhys = xnew;
        elseif ft == 2
            xPhys(:) = (H*xnew(:))./Hs;
        end
        if sum(xPhys(:)) > volfrac*nele, l1 = lmid; else l2 = lmid; end
    end
    pol = max(0,max(pol-move,min(1.,min(pol+move,sign(-dp))))); % OC update of polarization
    Density_change = max(abs(xnew(:)-x(:)));
    x = xnew;
    %% PLOT DENSITIES & POLARIZATION
    figure(1);colormap(gray); imagesc(1-[xPhys;flip(xPhys)]); caxis([0 1]); axis equal; axis off; drawnow;
    figure(2);colormap(jet); imagesc([((xPhys.*(pol*2-1))+1)/2;((flip(xPhys).*flip(pol*2-1))+1)/2]);  caxis([0 1]); axis equal; axis off; drawnow;
    fprintf(' It:%2.0i   Time:%3.2fs   Obj:%3.3f   Vol:%3.3f   ch:%3.3f\n ',loop,toc,c,mean(xPhys(:)),Density_change);
end
% ||=====================================================================||
% || THIS CODE IS WRITTEN BY ABBAS HOMAYOUNI-AMLASHI, THOMAS SCHLINQUER, ||
% ||    ABDENBI MOHAND-OUSAID AND MICKY RAKOTONDRABE                     ||
% ||=====================================================================||