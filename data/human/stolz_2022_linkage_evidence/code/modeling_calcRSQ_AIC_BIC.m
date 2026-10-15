% #########################################################################
% Calculate model fit indices
% Pseudo R^2, BIC, & AIC
% 
% Reward and punishment avoidance learning
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

clear

cd('C:\Users\cstolz\Documents\RevEx1_EEG_Export_BIDS\BIDS_EXPORT\code\');
currdir = pwd;

datapath = strcat(currdir,'\Data');
savepath = strcat(currdir,'\Saves');
cd(savepath);

% trial count
trials = 280;

% action selection options
choice_options = 2;

% model fit parameters, k = free parameter count
for model = 1:5
    if model == 1
        load('RevEx1_fmcon_M1_vanilla.mat'); % M1: Vanilla
        k = 2;
    elseif model == 2
        load('RevEx1_fmcon_M2_Kdupdate.mat'); % M2: Dual updating
        k = 3;
    elseif model == 3
        load('RevEx1_fmcon_M3_timeDecay.mat'); % M3: Time
        k = 3;      
    elseif model == 4
        load('RevEx1_fmcon_M4_decay.mat'); % M4: Decay non-chosen positive feedback
        k = 3;
    elseif model == 5
        load('RevEx1_fmcon_M5_decayAll.mat'); % M5: Decay non-chosen all trials
        k = 3;
    end
    
    % single subject
    % pseudo-R^2
    subjects = length(REW_LL);
    chanceprobability = 1/choice_options;
    LLE_templ=abs(sum(log(chanceprobability*ones(1,trials)))); % LLE random choice 

    for subj = 1:subjects
        REW_pseudoRsqr(subj,model)=mean((LLE_templ-REW_LL(subj,model))./LLE_templ);
        PUN_pseudoRsqr(subj,model)=mean((LLE_templ-PUN_LL(subj,model))./LLE_templ);
    end

    % AIC
    for subj = 1:subjects
        REW_AIC(subj,model)= -2*REW_LL(subj,model)+2*(k);
        PUN_AIC(subj,model)= -2*PUN_LL(subj,model)+2*(k);
    end

    % BIC
    for subj = 1:subjects
        REW_BIC(subj,model)= -2*REW_LL(subj,model)+log(trials)*k;
        PUN_BIC(subj,model)= -2*PUN_LL(subj,model)+log(trials)*k;
    end
    
end

cd(savepath);
save('RevEx1_fitsRQS_AIC_BIC.mat','REW_pseudoRsqr','PUN_pseudoRsqr','REW_AIC','PUN_AIC','REW_BIC','PUN_BIC');


% average across subjects
for model = 1:5
   REW_avg_pRS(model) = mean(REW_pseudoRsqr(:,model));
   PUN_avg_pRS(model) = mean(PUN_pseudoRsqr(:,model));
    
   REW_avg_AIC(model) = mean(REW_AIC(:,model));
   PUN_avg_AIC(model) = mean(PUN_AIC(:,model));    
   
   REW_avg_BIC(model) = mean(REW_BIC(:,model));
   PUN_avg_BIC(model) = mean(PUN_BIC(:,model)); 
   
   
   REW_std_pRS(model) = std(REW_pseudoRsqr(:,model));
   PUN_std_pRS(model) = std(PUN_pseudoRsqr(:,model));
    
   REW_std_AIC(model) = std(REW_AIC(:,model));
   PUN_std_AIC(model) = std(PUN_AIC(:,model));    
   
   REW_std_BIC(model) = std(REW_BIC(:,model));
   PUN_std_BIC(model) = std(PUN_BIC(:,model)); 
end

% find best fitting model
   disp('Best fitting model:');
   disp('REW pseudo-R^2');
   find(REW_avg_pRS == max(REW_avg_pRS))
   disp('PUN pseudo-R^2');   
   find(PUN_avg_pRS == max(PUN_avg_pRS))
   
   disp('REW AIC');   
   find(abs(REW_avg_AIC) == min(abs(REW_avg_AIC)))
   
   disp('PUN AIC');      
   find(abs(PUN_avg_AIC) == min(abs(PUN_avg_AIC)))
   
   disp('REW BIC');      
   find(abs(REW_avg_BIC) == min(abs(REW_avg_BIC)))
   
   disp('PUN BIC');      
   find(abs(PUN_avg_BIC) == min(abs(PUN_avg_BIC)))