%% Stochastic gradient on least squares problems
clc; clear all; close all;
%% Global parameters
global n L xin xbdc X nor yin act_fn betai betao dim;
%% Network parameters
n      = [4 30 1];               % dimension of each layer
m      = 6^3;                    % number of training data
mb     = ceil(m^(2/3));          % number of boundary training data
mk     = 3*mb;                   % number of interface data
m_test = 1d3*m;                  % number of test points
L      = length(n);              % number of total layers
dim    = n(1)-1;                 % problem dimension   
act_fn = "logsig";               % activation function
totWb  = dot(n(1:L-1)+1,n(2:L)); % total number of weights and biases
%% LM parameters
mu     = 1d+5;                   % damping parameter
maxit  = 1d3;                    % maximum number of epochs
TOL    = 5d-11;                  % tolerance
epoch  = 1;                      % initial step
p0     = randn(totWb,1);         % initial guess 
%% Problem setup
% ... domain boundary ...
bdA = -1; bdB = 1; bdC = -1; bdD = 1; bdE = -1; bdF = 1;
% ... contrast ratio ...
betai = 1d0; betao = 1d-3;
% ... exact solution ...
ui = @(x,y,z) exp(x+y+z);
uo = @(x,y,z) sin(x).*sin(y).*sin(z);
u  = @(x,y,z,w) (1-w)/2.*ui(x,y,z) + (1+w)/2.*uo(x,y,z);
grad_ui = @(x,y,z) [exp(x+y+z); exp(x+y+z); exp(x+y+z)];
grad_uo = @(x,y,z) [cos(x).*sin(y).*sin(z); sin(x).*cos(y).*sin(z); sin(x).*sin(y).*cos(z)];
% ... rhs function ...
fi   = @(x,y,z) 3*ui(x,y,z);
fo   = @(x,y,z) -3*uo(x,y,z);
f    = @(x,y,z,w) (w-1)/-2.*fi(x,y,z) + (w+1)/2.*fo(x,y,z);
% ... interface parameters ...
ra = .7; rb = .5; rc = .3;
% ... level set & feature functions ...
phi = @(x,y,z) (x/ra).^2 + (y/rb).^2+(z/rc).^2 - 1;
psi = @(x,y,z) hardlims(phi(x,y,z));
%% Given training data
% ... in the interior domain ...
x = .5*(bdA+bdB)+.5*(bdB-bdA)*cos((2*(1:m^(1/3))-1)/(2*m^(1/3))*pi);
y = .5*(bdC+bdD)+.5*(bdD-bdC)*cos((2*(1:m^(1/3))-1)/(2*m^(1/3))*pi);
z = .5*(bdE+bdF)+.5*(bdF-bdE)*cos((2*(1:m^(1/3))-1)/(2*m^(1/3))*pi);
xx = repmat(x',1,m^(1/3),m^(1/3));
yy = repmat(y ,m^(1/3),1,m^(1/3));
zz = reshape(repmat(z,m^(1/3)*m^(1/3),1),[m^(1/3),m^(1/3),m^(1/3)]);
xin      = [xx(:)'; yy(:)'; zz(:)'];
xin      = [xin; psi(xin(1,:),xin(2,:),xin(3,:))];
% ... on the boundary ...
xbdc     = [[ones(1,mb)*bdA ones(1,mb)*bdB;       repmat(reshape(yy(1,:,:),1,mb),1,2); repmat(reshape(zz(1,:,:),1,mb),1,2)   ] ...
            [repmat(reshape(xx(:,1,:),1,mb),1,2); ones(1,mb)*bdC ones(1,mb)*bdD;       repmat(reshape(zz(:,1,:),1,mb),1,2)   ] ...
            [repmat(reshape(xx(:,:,1),1,mb),1,2); repmat(reshape(yy(:,:,1),1,mb),1,2); ones(1,mb)*bdE ones(1,mb)*bdF]];
xbdc     = [xbdc; psi(xbdc(1,:),xbdc(2,:),xbdc(3,:))];
% ... on the interface ...
theta    = 2*pi*rand(1,mk); nu = asin(2*rand(1,mk)-1);
X        = [ra*cos(nu).*cos(theta); rb*cos(nu).*sin(theta); rc*sin(nu)];
grad_phi = [2*X(1,:)/ra^2; 2*X(2,:)/rb^2; 2*X(3,:)/rc^2];
nor      = grad_phi./vecnorm(grad_phi);
% % ... jump conditions ...
uj       = uo(X(1,:),X(2,:),X(3,:)) - ui(X(1,:),X(2,:),X(3,:));
duj      = sum( (betao*grad_uo(X(1,:),X(2,:),X(3,:))-betai*grad_ui(X(1,:),X(2,:),X(3,:))).*nor );
% % ... target output ...
yin  = [ f(xin(1,:),xin(2,:),xin(3,:),xin(4,:)) / sqrt(length(xin))      ...
         u(xbdc(1,:),xbdc(2,:),xbdc(3,:),xbdc(4,:)) / sqrt(length(xbdc)) ...
         uj / sqrt(length(X))                                            ...
         duj / sqrt(length(X)) ];    
% plot3(xin(1,:),xin(2,:),xin(3,:),'.',xbdc(1,:),xbdc(2,:),xbdc(3,:),'o'); axis equal; return;
% plot3(X(1,:),X(2,:),X(3,:),'.'); axis equal; hold on; quiver3(X(1,:),X(2,:),X(3,:),nor(1,:),nor(2,:),nor(3,:)); return;
%% Display network parameters
disp(['Training points : ', num2str(length(yin))]);
disp(['Parameters      : ', num2str(totWb)]);
%% Preallocation
a      = cell(L,1);                   % activation
loss   = zeros(maxit,1);              % loss function
mu_rec = zeros(maxit,1);              % record damping parameter
%% Training
tic;
while ( epoch <= maxit )
   % ... Computation of Jacobian matrix ...
   J = Jacobian_cost(p0);
   % ... Computation of the vector loss function ...
   res = cost_vec(p0);
   % ... Update p_{k+1} using Levenberg-Marquardt algorithm ...
   [U,S,V] = svd(J,"econ");
   p = p0 + V*(U'.*(diag(S)./(diag(S).^2+mu)))*res';  
   % ... Compute the loss function ...
   loss(epoch) = sum(res.^2);
   % ... Damping parameter strategy ...
   mu_rec(epoch) = mu;
   if mod(epoch,2) == 0 && loss(epoch)<loss(epoch-1), mu = max(mu/2,1d-09); end
   if mod(epoch,2) == 0 && loss(epoch)>loss(epoch-1), mu = min(mu*2,1d+08); end
   % ... Break the loop if the certain condition is satisfied ...
   if loss(epoch) <= TOL, break; end
   % ... Next iteration loop ...
   epoch = epoch+1; p0 = p;
end
loss( loss == 0 ) = []; mu_rec( mu_rec == 0 ) = [];
disp(['LOSS            : ', num2str(loss(end),'%.3e'),' (epoch = ', num2str(epoch),')']);
toc;
%% Testing & Output
% ... Output the training history ...
figure(1); loglog(loss,'linewidth',2); 
set(gca,'fontsize',20,'linewidth',1); title(['loss = ', num2str(loss(end),'%.3e')]); xlabel('training step'); grid on;
% ... Convert solution p to weights and biases ...
[W,b] = Param_2_Wb(p,n);
% ... Set test points & exact solution ...
x_test = [(bdB-bdA)*rand(1,m_test)+bdA;(bdD-bdC)*rand(1,m_test)+bdC;(bdF-bdE)*rand(1,m_test)+bdE];
a{1}   = [x_test; psi(x_test(1,:),x_test(2,:),x_test(3,:))];
u_test = u(a{1}(1,:),a{1}(2,:),a{1}(3,:),a{1}(4,:)); 
% ... Forward pass ...
y_test = W{3}*activation( W{2}*a{1}+b{2}, act_fn ) + b{3};
% ... Output ...
% disp(['L_inf error     : ', num2str( norm(y_test-u_test,inf),'%.3e' )]);
% disp(['L_2   error     : ', num2str( norm(y_test-u_test,2)/sqrt(length(y_test)),'%.3e' )]);

disp([num2str( norm(y_test-u_test,inf),'%.3e' )]);
disp([num2str( norm(y_test-u_test,2)/sqrt(length(y_test)),'%.3e' )]);

% scrsz = get(0,'ScreenSize');
% figure('Position',[250 10 scrsz(3)/2.5 scrsz(4)/2]);
% scatter3(x_test(1,:),x_test(2,:),x_test(3,:),30*abs(y_test-u_test)/norm(y_test-u_test,inf),abs(y_test-u_test),'filled');
% axis equal; set(gca,'linewidth',2,'fontsize',16); title('|\phi_S-\phi|'); colorbar;
%% Cost vector function
function f = cost_vec(p)
   global n L xin xbdc X nor yin act_fn betai betao dim;
   % ... preallocation ...
   a = cell(L,1); % activation
   % ... convert parameter to weights and biases
   [W,b] = Param_2_Wb(p,n);
   % ... forward pass : laplace(u) ...
   a{1} = xin;  
   LaplaceU = W{3}.*W{2}.^2'*d2activation( W{2}*a{1}+b{2}, act_fn );
   LaplaceU = sum( LaplaceU(1:dim,:) );
   % ... forward pass : u at boundary ...
   a{1} = xbdc;
   Ub = W{3}*activation( W{2}*a{1}+b{2}, act_fn ) + b{3};
   % ... forward pass : [u] at interface ...
   mk   = length(X);
   a{1} = [ [X;ones(1,mk)],[X;-ones(1,mk)] ];
      % ... [U] ...
      Uj = W{3}*activation( W{2}*a{1} + b{2}, act_fn ) + b{3};
      Uj = Uj(1:mk) - Uj(mk+1:2*mk);
      % ... [beta*dU/dn] ...
      dUj = W{3}.*W{2}'*dactivation( W{2}*a{1}+b{2}, act_fn );
      dUj = sum( (betao*dUj(1:dim,1:mk) - betai*dUj(1:dim,mk+1:2*mk)).*nor );
   % ... output ...
   f = yin - [LaplaceU/sqrt(length(xin)) Ub/sqrt(length(xbdc)) ...
              Uj/sqrt(length(X)) dUj/sqrt(length(X))];
end
%% Jacobian of cost function
function J = Jacobian_cost(p)
   global n xin xbdc X nor act_fn betai betao dim;
   % ... convert parameter to weights and biases ...
   [W,b] = Param_2_Wb(p,n);
   % ... grad of loss vector of Laplace(u) ...
   a{1}  = xin;
   z{2}  = W{2}*a{1}+b{2};
   d2a   = d2activation( z{2}, act_fn );
   d3a   = d3activation( z{2}, act_fn );
   
   dLaplaceU_b{2} = W{3}.*sum(W{2}(:,1:dim).^2,2)'.* d3a';
   dLaplaceU_b{3} = zeros(length(a{1}),1);
   dLaplaceU_W{2} = [reshape(W{2}(:,1:dim),1,dim*n(2)).*repmat(2*W{3}.*d2a',1,dim) zeros(length(a{1}),n(2))] + ...
                    repmat(dLaplaceU_b{2},1,dim+1).*kron(a{1}',ones(1,n(2)));
   dLaplaceU_W{3} = sum(W{2}(:,1:dim).^2,2)'.* d2a';
   % ... grad of loss vector of u at boundary ...
   a{1} = xbdc;
   z{2} = W{2}*a{1}+b{2};
   a{2} = activation( z{2}, act_fn );
   da   = dactivation( z{2}, act_fn );

   dUb_b{2} = W{3}.* da';
   dUb_b{3} = ones(length(a{1}),1);
   dUb_W{2} = repmat(dUb_b{2},1,dim+1).*kron(a{1}',ones(1,n(2)));
   dUb_W{3} = a{2}';
   % ... grad of loss vector of jump conditions at interface ...
   mk   = length(X);
   a{1} = [ [X;-ones(1,mk)],[X;ones(1,mk)] ];
   z{2} = W{2}*a{1}+b{2};
   a{2} = activation( z{2}, act_fn );
   da   = dactivation( z{2}, act_fn );
   d2a  = d2activation( z{2}, act_fn );
      % ... [U] ...
      dUj_b{2} = W{3}.* da';
      dUj_b{3} = ones(length(a{1}),1);
      dUj_W{2} = repmat(dUj_b{2},1,dim+1).*kron(a{1}',ones(1,n(2)));
      dUj_W{3} = a{2}';
      
      dUj_b{2} = dUj_b{2}(mk+1:2*mk,:) - dUj_b{2}(1:mk,:);
      dUj_b{3} = dUj_b{3}(mk+1:2*mk,:) - dUj_b{3}(1:mk,:);
      dUj_W{2} = dUj_W{2}(mk+1:2*mk,:) - dUj_W{2}(1:mk,:);
      dUj_W{3} = dUj_W{3}(mk+1:2*mk,:) - dUj_W{3}(1:mk,:);
      % ... [beta*dU/dn] ...
      dUx = mat2cell( reshape(W{2}(:,1:dim),1,dim*n(2)).*repmat(W{3}.*d2a',1,dim), length(a{1}), n(2)*ones(1,dim) );
      ddUj_b{2} = sum( reshape( cell2mat(dUx).*kron([nor nor]',ones(1,n(2))), length(a{1}),n(2),dim ), 3 );  
      ddUj_b{3} = zeros(length(a{1}),1);
      ddUj_W{2} = repmat(ddUj_b{2},1,dim+1).*kron(a{1}',ones(1,n(2))) + ...
                  [repmat(W{3}.*da',1,dim).*kron([nor nor]',ones(1,n(2))) zeros(length(a{1}),n(2))];
      ddUj_W{3} = sum( reshape( (repmat(da',1,dim).*reshape(W{2}(:,1:dim),1,dim*n(2))).*kron([nor nor]',ones(1,n(2))), length(a{1}),n(2),dim ), 3);
      
      ddUj_b{2} = betao*ddUj_b{2}(mk+1:2*mk,:) - betai*ddUj_b{2}(1:mk,:);
      ddUj_b{3} = betao*ddUj_b{3}(mk+1:2*mk,:) - betai*ddUj_b{3}(1:mk,:);
      ddUj_W{2} = betao*ddUj_W{2}(mk+1:2*mk,:) - betai*ddUj_W{2}(1:mk,:);
      ddUj_W{3} = betao*ddUj_W{3}(mk+1:2*mk,:) - betai*ddUj_W{3}(1:mk,:);
   % ... output ...
   J = [ [dLaplaceU_W{2} dLaplaceU_b{2} dLaplaceU_W{3} dLaplaceU_b{3}]/sqrt(length(xin)) ; ...
         [dUb_W{2}       dUb_b{2}       dUb_W{3}       dUb_b{3}      ]/sqrt(length(xbdc)); ...
         [dUj_W{2}       dUj_b{2}       dUj_W{3}       dUj_b{3}      ]/sqrt(length(X))   ; ...
         [ddUj_W{2}      ddUj_b{2}      ddUj_W{3}      ddUj_b{3}     ]/sqrt(length(X))  ];
end
%% Activation function
function y = activation(x,act_fn)
   if strcmp(act_fn,'logsig'),  y = logsig(x);  end
   if strcmp(act_fn,'tansig'),  y = tansig(x);  end
   if strcmp(act_fn,'poslin'),  y = poslin(x);  end
   if strcmp(act_fn,'purelin'), y = purelin(x); end
end
function y = dactivation(x,act_fn)
   if strcmp(act_fn,'logsig'),  y = logsig(x).*(1-logsig(x)); end
   if strcmp(act_fn,'tansig'),  y = 1-tansig(x).^2;           end
   if strcmp(act_fn,'poslin'),  y = ones(size(x));            end
   if strcmp(act_fn,'purelin'), y = ones(size(x));            end
end
function y = d2activation(x,act_fn)
   if strcmp(act_fn,'logsig'),  y = logsig(x).*(1-logsig(x)).*(1-2*logsig(x)); end
   if strcmp(act_fn,'tansig'),  y = (1-tansig(x).^2).*(-2*tansig(x));          end
   if strcmp(act_fn,'poslin'),  y = zeros(size(x));                            end
   if strcmp(act_fn,'purelin'), y = zeros(size(x));                            end
end
function y = d3activation(x,act_fn)
   if strcmp(act_fn,'logsig'),  y = logsig(x).*(1-logsig(x)).*(1-6*logsig(x)+6*logsig(x).^2); end
   if strcmp(act_fn,'tansig'),  y = (1-tansig(x).^2).*(-2+6*tansig(x).^2);                    end
   if strcmp(act_fn,'poslin'),  y = zeros(size(x));                                           end
   if strcmp(act_fn,'purelin'), y = zeros(size(x));                                           end
end