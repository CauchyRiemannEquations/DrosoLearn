% #########################################################################
% Q learning function for models with different parameter in reward gain
% and punishment avoidance reinforcement learning
% 
% M1: Vanilla
% M2: Dual updating
% M3: Time
% M4: Decay non-chosen positive feedback
% M5: Decay non-chosen all trials
%
% Author: Christopher Stolz, Leibniz Institute for Neurobiology, 
% Magdeburg, Germany
% Contact: stolz.unrast@gmail.com
% Version 1: October 14th 2022
% #########################################################################

function [neg_LLE Q_gen pr_gen Qval] = Q_learning(parameter,model,a,r,context,t)

% init
% parameters
beta = parameter(1); % beta, temperature
alpha = parameter(2); % alpha, learning rate
decay = parameter(3); % decay parameter
k = parameter(4); % dual update
lambda = parameter(5); % time decay

% LLE
LLE = 0;

% context definition: reward gain & punishment avoidance
if context == 1 % reward gain
    Q(1:length(unique(a)))= 0;
    ctxt_par = 0;
elseif context == 2 % punishment avoidance
    Q(1:length(unique(a)))= -1;
    ctxt_par = 1;
    r = r - 1;
end

r_outcomes = unique(r);

for i = 1:length(a)
    
    P(a(i)) = (exp(beta*Q(a(i)))/sum(exp(beta*Q(:))));
    LLE = LLE + log(P(a(i)));
     
    % prediction error
    if model == 1 % M1: Vanilla
        Q_gen(i) = Q(a(i)); % general Q value, chosen option  

        for p = 1:length(unique(a)) % Q values for both options, for return
            Qval(p,i) = Q(p);
        end        
        
        pr(a(i)) =  r(i) - Q(a(i)); % current prediction error   
        pr_gen(i) = pr(a(i)); % prediction error for return variable

        Q(a(i)) = Q(a(i)) + alpha * pr(a(i)); % current Q value
                
    elseif model == 2 % M2: Dual updating
        Q_gen(i) = Q(a(i)); % general Q value, chosen option    

        for p = 1:length(unique(a)) % Q values for both options, for return
            Qval(p,i) = Q(p);
        end        

        pr(a(i)) =  r(i) - Q(a(i)); % current prediction error 
        pr_gen(i) = pr(a(i)); % prediction error for return variable
        
        Q(a(i)) = Q(a(i)) + alpha * pr(a(i)); % current Q value
               
        % dual updating
        if r(i) == r_outcomes(1)
            r_f = r_outcomes(2);
        else
            r_f = r_outcomes(1);
        end
        
        for g = 1:length(unique(a))
            if g ~= a(i)
                Q(g) = Q(g) + k*alpha*(r_f-Q(g));  % Q value
            end
        end 
        
    elseif model == 3 % M3: Time
        Q_gen(i) = Q(a(i)); % general Q value, chosen option  

        for p = 1:length(unique(a)) % Q values for both options, for return
            Qval(p,i) = Q(p);
        end               

        pr(a(i)) =  r(i) - Q(a(i)); % current prediction error 
        pr_gen(i) = pr(a(i)); % prediction error for return variable
        
        Q(a(i)) = Q(a(i)) + alpha * (lambda*1/t(i))* pr(a(i)); % current Q value
        
    elseif model == 4 % M4: Decay non-chosen positive feedback
        Q_gen(i) = Q(a(i)); % general Q value, chosen option       

        for p = 1:length(unique(a)) % Q values for both options, for return
            Qval(p,i) = Q(p);
        end        

        pr(a(i)) =  r(i) - Q(a(i)); % current prediction error 
        pr_gen(i) = pr(a(i)); % prediction error for return variable
        
        Q(a(i)) = Q(a(i)) + alpha * pr(a(i)); % current Q value
        
        % update of non-chosen option, positive feedback, decay parameter
        if r(i) == (1-ctxt_par) % outcome, 1 = gain
            for g = 1:length(unique(a))
                if g ~= a(i)
                    if context == 1
                        Q(g) = (1-decay)*Q(g);
                    elseif context == 2
                        Q(g) = (1-decay)*(1+Q(g))-1;
                    end
                end
            end   
        end
        
    elseif model == 5 % M5: Decay non-chosen all trials
        Q_gen(i) = Q(a(i)); % general Q value, chosen option

        for p = 1:length(unique(a)) % Q values for both options, for return
            Qval(p,i) = Q(p);
        end        

        pr(a(i)) =  r(i) - Q(a(i)); % current prediction error   
        pr_gen(i) = pr(a(i)); % prediction error for return variable
        
        Q(a(i)) = Q(a(i)) + alpha * pr(a(i)); % current Q value
        
        % update of non-chosen option, decay parameter
        for g = 1:length(unique(a))
            if g ~= a(i)
                if context == 1
                    Q(g) = (1-decay)*Q(g);
                elseif context == 2
                    Q(g) = (1-decay)*(1+Q(g))-1;
                end
            end
        end
        
    end;
    
    
end

neg_LLE = -LLE; 
