%% Stochastic gradient on least squares problems
clc; clear all; close all;
%% Global parameters
global n L xin xbdc X nor yin act_fn betai betao dim;
%% Network parameters
n      = [7 50 1];                    % dimension of each layer
m      = 100;                         % number of training data
mb     = 3*ceil(m^((n(1)-2)/(n(1)-1))); % number of boundary training data
mk     = 3*ceil(m^((n(1)-2)/(n(1)-1))); % number of interface data
m_test = 1d3*m;                       % number of testing points
L      = length(n);                   % number of total layers
dim    = n(1)-1;                      % problem dimension   
act_fn = "logsig";                    % activation function
totWb  = dot(n(1:L-1)+1,n(2:L));      % total number of weights and biases
%% LM parameters
mu     = 1d+0;                        % damping parameter
maxit  = 1d3;                         % maximum number of epochs
TOL    = 5d-10;                       % tolerance
epoch  = 1;                           % initial step
p0     = randn(totWb,1);              % initial guess 
%% Problem setup
% ... domain boundary ...
Ra = .6; Rb = .6; Rc = .6; Rd = .6; Re = .6; Rf = .6;
ra = .5; rb = .5; rc = .5; rd = .5; re = .5; rf = .5;
% ... contrast ratio ...
betai = 1d0; betao = 1d-3;
% ... exact solution ...
ui = @(x,y,z,u,v,w) exp(x+y+z+u+v+w);
uo = @(x,y,z,u,v,w) sin(x).*sin(y).*sin(z).*sin(u).*sin(v).*sin(w);
ue = @(x,y,z,u,v,w,id) (1-id)/2.*ui(x,y,z,u,v,w) + (1+id)/2.*uo(x,y,z,u,v,w);
grad_ui = @(x,y,z,u,v,w) [exp(x+y+z+u+v+w); exp(x+y+z+u+v+w); exp(x+y+z+u+v+w); ...
                          exp(x+y+z+u+v+w); exp(x+y+z+u+v+w); exp(x+y+z+u+v+w)];
grad_uo = @(x,y,z,u,v,w) [cos(x).*sin(y).*sin(z).*sin(u).*sin(v).*sin(w); ...
                          sin(x).*cos(y).*sin(z).*sin(u).*sin(v).*sin(w); ...
                          sin(x).*sin(y).*cos(z).*sin(u).*sin(v).*sin(w); ...
                          sin(x).*sin(y).*sin(z).*cos(u).*sin(v).*sin(w); ...
                          sin(x).*sin(y).*sin(z).*sin(u).*cos(v).*sin(w); ...
                          sin(x).*sin(y).*sin(z).*sin(u).*sin(v).*cos(w)];
% ... rhs function ...
fi   = @(x,y,z,u,v,w) dim*ui(x,y,z,u,v,w);
fo   = @(x,y,z,u,v,w) -dim*uo(x,y,z,u,v,w);
f    = @(x,y,z,u,v,w,id) (id-1)/-2.*fi(x,y,z,u,v,w) + (id+1)/2.*fo(x,y,z,u,v,w);
% .. level set & feature functions ...
phi = @(x,y,z,u,v,w) (x/ra).^2+(y/rb).^2+(z/rc).^2+(u/rd).^2+(v/re).^2+(w/rf).^2-1;
psi = @(x,y,z,u,v,w) hardlims(phi(x,y,z,u,v,w));
%% Given training data
% ... in the interior domain ... 
theta = 2*pi*rand(1,m); nu1 = acos(2*rand(1,m)-1); nu2 = acos(2*rand(1,m)-1); nu3 = acos(2*rand(1,m)-1); nu4 = acos(2*rand(1,m)-1); rad = rand(1,m).^(1/dim);
xin(1,:) = Ra*rad.*cos(nu1);
xin(2,:) = Rb*rad.*sin(nu1).*cos(nu2);
xin(3,:) = Rc*rad.*sin(nu1).*sin(nu2).*cos(nu3);
xin(4,:) = Rd*rad.*sin(nu1).*sin(nu2).*sin(nu3).*cos(nu4);
xin(5,:) = Re*rad.*sin(nu1).*sin(nu2).*sin(nu3).*sin(nu4).*cos(theta);
xin(6,:) = Rf*rad.*sin(nu1).*sin(nu2).*sin(nu3).*sin(nu4).*sin(theta);  
xin      = [xin; psi(xin(1,:),xin(2,:),xin(3,:),xin(4,:),xin(5,:),xin(6,:))];
% ... on the boundary ...
theta = 2*pi*rand(1,mb); nu1 = acos(2*rand(1,mb)-1); nu2 = acos(2*rand(1,mb)-1); nu3 = acos(2*rand(1,mb)-1); nu4 = acos(2*rand(1,mb)-1);
xbdc(1,:) = Ra*cos(nu1);
xbdc(2,:) = Rb*sin(nu1).*cos(nu2);
xbdc(3,:) = Rc*sin(nu1).*sin(nu2).*cos(nu3);
xbdc(4,:) = Rd*sin(nu1).*sin(nu2).*sin(nu3).*cos(nu4);
xbdc(5,:) = Re*sin(nu1).*sin(nu2).*sin(nu3).*sin(nu4).*cos(theta);
xbdc(6,:) = Rf*sin(nu1).*sin(nu2).*sin(nu3).*sin(nu4).*sin(theta);  
xbdc      = [xbdc; psi(xbdc(1,:),xbdc(2,:),xbdc(3,:),xbdc(4,:),xbdc(5,:),xbdc(6,:))];
% ... on the interface ...
theta = 2*pi*rand(1,mk); nu1 = acos(2*rand(1,mk)-1); nu2 = acos(2*rand(1,mk)-1); nu3 = acos(2*rand(1,mk)-1); nu4 = acos(2*rand(1,mk)-1);
X(1,:) = ra*cos(nu1);
X(2,:) = rb*sin(nu1).*cos(nu2);
X(3,:) = rc*sin(nu1).*sin(nu2).*cos(nu3);
X(4,:) = rd*sin(nu1).*sin(nu2).*sin(nu3).*cos(nu4);
X(5,:) = re*sin(nu1).*sin(nu2).*sin(nu3).*sin(nu4).*cos(theta);
X(6,:) = rf*sin(nu1).*sin(nu2).*sin(nu3).*sin(nu4).*sin(theta);
grad_phi = [2*X(1,:)/ra^2; 2*X(2,:)/rb^2; 2*X(3,:)/rc^2; 2*X(4,:)/rd^2; 2*X(5,:)/re^2; 2*X(6,:)/rf^2;];
nor      = grad_phi./vecnorm(grad_phi);
% % ... jump conditions ...
uj       = uo(X(1,:),X(2,:),X(3,:),X(4,:),X(5,:),X(6,:)) - ui(X(1,:),X(2,:),X(3,:),X(4,:),X(5,:),X(6,:));
duj      = sum( (betao*grad_uo(X(1,:),X(2,:),X(3,:),X(4,:),X(5,:),X(6,:))-betai*grad_ui(X(1,:),X(2,:),X(3,:),X(4,:),X(5,:),X(6,:))).*nor );
% % ... target output ...
yin  = [ f(xin(1,:),xin(2,:),xin(3,:),xin(4,:),xin(5,:),xin(6,:),xin(7,:)) / sqrt(length(xin))          ...
         ue(xbdc(1,:),xbdc(2,:),xbdc(3,:),xbdc(4,:),xbdc(5,:),xbdc(6,:),xbdc(7,:)) / sqrt(length(xbdc)) ...
         uj / sqrt(length(X))                                                                           ...
         duj / sqrt(length(X)) ];
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
theta = 2*pi*rand(1,m_test); nu1 = acos(2*rand(1,m_test)-1); nu2 = acos(2*rand(1,m_test)-1); nu3 = acos(2*rand(1,m_test)-1); nu4 = acos(2*rand(1,m_test)-1); rad = rand(1,m_test).^(1/dim);
x_test(1,:) = Ra*rad.*cos(nu1);
x_test(2,:) = Rb*rad.*sin(nu1).*cos(nu2);
x_test(3,:) = Rc*rad.*sin(nu1).*sin(nu2).*cos(nu3);
x_test(4,:) = Rd*rad.*sin(nu1).*sin(nu2).*sin(nu3).*cos(nu4);
x_test(5,:) = Re*rad.*sin(nu1).*sin(nu2).*sin(nu3).*sin(nu4).*cos(theta);
x_test(6,:) = Rf*rad.*sin(nu1).*sin(nu2).*sin(nu3).*sin(nu4).*sin(theta);   
a{1}        = [x_test; psi(x_test(1,:),x_test(2,:),x_test(3,:),x_test(4,:),x_test(5,:),x_test(6,:))];
u_test      = ue(a{1}(1,:),a{1}(2,:),a{1}(3,:),a{1}(4,:),a{1}(5,:),a{1}(6,:),a{1}(7,:)); 
% ... forward pass ...
y_test      = W{3}*activation( W{2}*a{1}+b{2}, act_fn ) + b{3};
% ... output ...
disp(['L_inf error     : ', num2str( norm(y_test-u_test,inf),'%.3e' )]);
disp(['L_2   error     : ', num2str( norm(y_test-u_test,2)/sqrt(m_test),'%.3e' )]);
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
   f = yin - [LaplaceU/sqrt(length(xin)) Ub/sqrt(length(xbdc)) Uj/sqrt(length(X)) dUj/sqrt(length(X))];
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