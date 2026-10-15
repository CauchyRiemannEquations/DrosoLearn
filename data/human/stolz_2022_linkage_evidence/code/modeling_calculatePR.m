% #########################################################################
% Calculate single-trial prediction errors for further analyses
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


% define paths
datapath = strcat(currdir,'\Data');
savepath = strcat(currdir,'\Saves');
cd(savepath);


% load actions, feedback + time intervals
cd(datapath);
load('RevEx1_behavioralData.mat');
cd(currdir);

% derive single-trial prediction errors (PEs) and Q values for every model
for model = 1:5
    disp(num2str(model));
    cd(savepath);
    
    if model == 1
       load('RevEx1_fmcon_M1_vanilla.mat');
    elseif model == 2
       load('RevEx1_fmcon_M2_Kdupdate.mat');
    elseif model == 3
       load('RevEx1_fmcon_M3_timeDecay.mat');          
    elseif model == 4
       load('RevEx1_fmcon_M4_decay.mat');
    elseif model == 5
       load('RevEx1_fmcon_M5_decayAll.mat');   
    end;
        

   cd(currdir);
 

for subj = 1:length(REW_action)
    if model == 1
        [REW_neg_log_likelihood(subj) Q_REW_gen{subj} PR_REW_gen{subj} REW_Q{subj}]  = Q_learning([REW_beta_SM(subj) REW_alpha(subj) 0 0 0],model,REW_action{subj},REW_feedback{subj},1,REW_responseFB_interval{subj});
        [PUN_neg_log_likelihood(subj) Q_PUN_gen{subj} PR_PUN_gen{subj} PUN_Q{subj}]  = Q_learning([PUN_beta_SM(subj) PUN_alpha(subj) 0 0 0],model,PUN_action{subj},PUN_feedback{subj},1,PUN_responseFB_interval{subj});
    elseif model == 2
        [REW_neg_log_likelihood(subj) Q_REW_gen{subj} PR_REW_gen{subj} REW_Q{subj}]  = Q_learning([REW_beta_SM(subj) REW_alpha(subj) 0 REW_k_DU(subj) 0],model,REW_action{subj},REW_feedback{subj},1,REW_responseFB_interval{subj});
        [PUN_neg_log_likelihood(subj) Q_PUN_gen{subj} PR_PUN_gen{subj} PUN_Q{subj}]  = Q_learning([PUN_beta_SM(subj) PUN_alpha(subj) 0 PUN_k_DU(subj) 0],model,PUN_action{subj},PUN_feedback{subj},1,PUN_responseFB_interval{subj});
    elseif model == 3    
        [REW_neg_log_likelihood(subj) Q_REW_gen{subj} PR_REW_gen{subj} REW_Q{subj}]  = Q_learning([REW_beta_SM(subj) REW_alpha(subj) 0 0 REW_decayTime(subj)],model,REW_action{subj},REW_feedback{subj},1,REW_responseFB_interval{subj});
        [PUN_neg_log_likelihood(subj) Q_PUN_gen{subj} PR_PUN_gen{subj} PUN_Q{subj}]  = Q_learning([PUN_beta_SM(subj) PUN_alpha(subj) 0 0 PUN_decayTime(subj)],model,PUN_action{subj},PUN_feedback{subj},1,PUN_responseFB_interval{subj});
    elseif model == 4 || model == 5
        [REW_neg_log_likelihood(subj) Q_REW_gen{subj} PR_REW_gen{subj} REW_Q{subj}]  = Q_learning([REW_beta_SM(subj) REW_alpha(subj) REW_decay(subj) 0 0],model,REW_action{subj},REW_feedback{subj},1,REW_responseFB_interval{subj});
        [PUN_neg_log_likelihood(subj) Q_PUN_gen{subj} PR_PUN_gen{subj} PUN_Q{subj}]  = Q_learning([PUN_beta_SM(subj) PUN_alpha(subj) PUN_decay(subj) 0 0],model,PUN_action{subj},PUN_feedback{subj},1,PUN_responseFB_interval{subj});
    end
end

    savename = strcat('RevEx1_PR_M',num2str(model),'.mat');

    
    cd(savepath);
    save(savename,'PR_REW_gen','PR_PUN_gen','Q_REW_gen','Q_PUN_gen','REW_Q','PUN_Q');
    disp('finished');
end

