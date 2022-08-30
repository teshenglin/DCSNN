%% Stochastic gradient on least squares problems
clc; clear all; close all;
%% Global parameters
global n L xin xbdc X nor yin act_fn betai betao;
%% Network parameters
n      = [3 20 1];               % dimension of each layer
m      = 8^2;                    % number of training data
mb     = 4*ceil(sqrt(m));        % number of boundary training data
mk     = 4*ceil(sqrt(m));        % number of interface data
m_test = 1d3*m;                  % number of test data
L      = length(n);              % number of total layers
act_fn = "logsig";               % activation function
totWb  = dot(n(1:L-1)+1,n(2:L)); % total number of weights and biases
%% LM parameters
mu     = 1d+2;                   % damping parameter
maxit  = 2d3;                    % maximum number of epochs
TOL    = 1d-11;                  % tolerance
epoch  = 1;                      % initial step
p0     = randn(totWb,1);         % initial guess 
%% Problem setup
% ... domain boundary ...
ro = @(x)  1 - 0.3*cos(5*x);
% ... contrast ratio ...
betai = 1d0; betao = 1d-3;
% ... target function ...
ui   = @(x,y) exp(x+y);
uo   = @(x,y) sin(x).*sin(y);
u    = @(x,y,z) (z-1)/-2.*ui(x,y) + (z+1)/2.*uo(x,y);
grad_ui = @(x,y) [exp(x+y); exp(x+y)];
grad_uo = @(x,y) [cos(x).*sin(y); sin(x).*cos(y)];
% ... rhs function ...
fi   = @(x,y) 2*ui(x,y);
fo   = @(x,y) -2*uo(x,y);
f    = @(x,y,z) (z-1)/-2.*fi(x,y) + (z+1)/2.*fo(x,y);
% ... interface parameters ...
ri   = @(x) .4 - 0.2*cos(5*x);
dri  = @(x) sin(5*x);
% .. level set & feature functions ...
phi  = @(x,y) sqrt(x.^2+y.^2) - ri(atan2(y,x));
psi  = @(x,y) hardlims(phi(x,y));
%% Set training and target points
% ... in the interior domain ...
theta = sort(2*pi*rand(1,m)); scal = rand(1,m).^(1/2);
xin   = [ro(theta).*scal.*cos(theta); ro(theta).*scal.*sin(theta)];
xin   = [xin; psi(xin(1,:),xin(2,:))];
% ... on the domain boundary ...
theta  = 2*pi*rand(1,mb);
xbdc  = [ro(theta).*cos(theta); ro(theta).*sin(theta) ];
xbdc  = [xbdc; psi(xbdc(1,:),xbdc(2,:))];
% ... on the interface ...
theta  = 2*pi*rand(1,mk);
X(1,:) = ri(theta).*cos(theta); X(2,:) = ri(theta).*sin(theta);
nor    = [dri(theta).*sin(theta)+ri(theta).*cos(theta); ...
         -dri(theta).*cos(theta)+ri(theta).*sin(theta)];
nor    = nor./vecnorm(nor);
% ... jump conditions ...
uj   = uo(X(1,:),X(2,:)) - ui(X(1,:),X(2,:));
duj  = sum( (betao*grad_uo(X(1,:),X(2,:)) - betai*grad_ui(X(1,:),X(2,:))).*nor );
% ... target outputs ...
yin  = [ f(xin(1,:),xin(2,:),xin(3,:)) / sqrt(length(xin))     ...
         u(xbdc(1,:),xbdc(2,:),xbdc(3,:)) / sqrt(length(xbdc)) ...
         uj / sqrt(length(X))                                  ...
         duj / sqrt(length(X))];
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
   if mod(epoch,2) == 0 && loss(epoch)<loss(epoch-1), mu = max(mu/3,1d-09); end
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
theta  = sort(2*pi*rand(1,m_test)); scal = rand(1,m_test).^(1/2);
x_test = [ro(theta).*scal.*cos(theta); ro(theta).*scal.*sin(theta)];
a{1}   = [x_test; psi(x_test(1,:),x_test(2,:))];
u_test = u(a{1}(1,:),a{1}(2,:),a{1}(3,:));
% ... forward pass ...
y_test = W{3}*activation( W{2}*a{1}+b{2}, act_fn ) + b{3};
% ... Output ...
disp(['L_inf error     : ', num2str( norm(y_test(:)-u_test(:), inf), '%.3e' )]);
disp(['L_2   error     : ', num2str( norm(y_test(:)-u_test(:), 2)/sqrt(m_test), '%.3e' )]);

scrsz = get(0,'ScreenSize');
figure('Position',[250 10 scrsz(3)/1.7 scrsz(4)/2.7]);

subplot(1,2,1);
scatter3( x_test(1,:), x_test(2,:), y_test,25, y_test, 'filled');
set(gca,'linewidth',2,'fontsize',16); title('\phi_S'); xlabel('x_1'); ylabel('x_2');

subplot(1,2,2);
scatter3( x_test(1,:), x_test(2,:), abs(y_test-u_test), 25, abs(y_test-u_test), 'filled');
set(gca,'linewidth',2,'fontsize',16); title('|\phi_S-\phi|'); xlabel('x_1'); ylabel('x_2');
%% Cost vector function
function f = cost_vec(p)
   global n L xin xbdc X nor yin act_fn betai betao;
   % ... preallocation ...
   a = cell(L,1); % activation
   % ... convert parameter to weights and biases
   [W,b] = Param_2_Wb(p,n);
   % ... forward pass : laplace(u) ...
   a{1} = xin; 
   LaplaceU = W{3}.*W{2}.^2'*d2activation( W{2}*a{1}+b{2}, act_fn );     
   LaplaceU = sum( LaplaceU(1:2,:) );
   % ... forward pass : u at boundary ...
   a{1}  = xbdc;
   Ub    = W{3}*activation( W{2}*a{1}+b{2}, act_fn ) + b{3};
   % ... forward pass : [u] at interface ...
   mk    = length(X);
   a{1}  = [ [X;ones(1,mk)],[X;-ones(1,mk)] ];
   z{2}  = W{2}*a{1}+b{2};
   gradU = W{3}.*W{2}'*dactivation( z{2}, act_fn );
      % ... [U] ...
      Uj = W{3}*activation( z{2}, act_fn ) + b{3};
      Uj = Uj(1:mk) - Uj(mk+1:2*mk);
      % ... [beta*dU/dn] ...
      dUdx = gradU(1,:);
      dUdy = gradU(2,:);
      dUj  = ( betao*dUdx(1:mk) - betai*dUdx(mk+1:2*mk) ).*nor(1,:) + ...
             ( betao*dUdy(1:mk) - betai*dUdy(mk+1:2*mk) ).*nor(2,:);
   % ... output ...
   f = yin - [LaplaceU/sqrt(length(xin)) Ub/sqrt(length(xbdc)) ...
              Uj/sqrt(length(X)) dUj/sqrt(length(X))];
end
%% Jacobian of cost function
function J = Jacobian_cost(p)
   global n xin xbdc X nor act_fn betai betao;
   % ... convert parameter to weights and biases ...
   [W,b] = Param_2_Wb(p,n);
   % ... grad of loss vector of Laplace(u) ...
   a{1}  = xin;
   z{2}  = W{2}*a{1}+b{2};
   d2a   = d2activation( z{2}, act_fn );
   d3a   = d3activation( z{2}, act_fn );
   
   dLaplaceU_b{2} = W{3}.* ( W{2}(:,1).^2' + W{2}(:,2).^2' ) .* d3a';
   dLaplaceU_b{3} = zeros(length(a{1}),1);
   dLaplaceU_W{2} = [ 2*W{3}.*W{2}(:,1)'.* d2a' + dLaplaceU_b{2} .* a{1}(1,:)' ...
                      2*W{3}.*W{2}(:,2)'.* d2a' + dLaplaceU_b{2} .* a{1}(2,:)' ...
                      dLaplaceU_b{2} .* a{1}(3,:)'];
   dLaplaceU_W{3} = ( W{2}(:,1).^2' + W{2}(:,2).^2' ).* d2a';
   % ... grad of loss vector of u at boundary ...
   a{1} = xbdc;
   z{2} = W{2}*a{1}+b{2};
   a{2} = activation( z{2}, act_fn );
   da   = dactivation( z{2}, act_fn );

   dUb_b{2} = W{3}.* da';
   dUb_b{3} = ones(length(a{1}),1);
   dUb_W{2} = [ dUb_b{2}.*a{1}(1,:)' dUb_b{2}.*a{1}(2,:)' dUb_b{2}.*a{1}(3,:)'];
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
      dUj_W{2} = [ dUj_b{2}.*a{1}(1,:)' dUj_b{2}.*a{1}(2,:)' dUj_b{2}.*a{1}(3,:)'];
      dUj_W{3} = a{2}';
      
      dUj_b{2} = dUj_b{2}(mk+1:2*mk,:) - dUj_b{2}(1:mk,:);
      dUj_b{3} = dUj_b{3}(mk+1:2*mk,:) - dUj_b{3}(1:mk,:);
      dUj_W{2} = dUj_W{2}(mk+1:2*mk,:) - dUj_W{2}(1:mk,:);
      dUj_W{3} = dUj_W{3}(mk+1:2*mk,:) - dUj_W{3}(1:mk,:);
      % ... [beta*dU/dn] ...
      dUx1_b{2} = W{3}.*W{2}(:,1)'.*d2a';
      dUx2_b{2} = W{3}.*W{2}(:,2)'.*d2a';
      ddUj_b{2} = dUx1_b{2}.*[nor(1,:)'; nor(1,:)'] + dUx2_b{2}.*[nor(2,:)'; nor(2,:)'];
      ddUj_b{3} = zeros(length(a{1}),1);
      ddUj_W{2} = [ (W{3}.*da'+dUx1_b{2}.*a{1}(1,:)').*[nor(1,:)'; nor(1,:)'] + (dUx2_b{2}.*a{1}(1,:)').*[nor(2,:)'; nor(2,:)'] ...
                    (dUx1_b{2}.*a{1}(2,:)').*[nor(1,:)'; nor(1,:)'] + (W{3}.*da'+dUx2_b{2}.*a{1}(2,:)').*[nor(2,:)'; nor(2,:)'] ...
                    (dUx1_b{2}.*a{1}(3,:)').*[nor(1,:)'; nor(1,:)'] + (dUx2_b{2}.*a{1}(3,:)').*[nor(2,:)'; nor(2,:)']];
      ddUj_W{3} = (W{2}(:,1)'.*da').*[nor(1,:)'; nor(1,:)'] + (W{2}(:,2)'.*da').*[nor(2,:)'; nor(2,:)'];
      
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