% #########################################################################
% Model fit of Q learning models with different parameter for reward gain
% and punishment avoidance reinforcement learning
% 
% M1: Vanilla
% M2: Dual updating
% M3: Time
% M4: Decay non-chosen positive feedback
% M5: Decay non-chosen all trials
%
% Fit options based on Palminteri, Khamassi, Joffily, & Coricelli (2015),
% Nature Communications
%
% Author: Christopher Stolz, Leibniz Institute for Neurobiology, 
% Magdeburg, Germany
% Contact: stolz.unrast@gmail.com
% Version 1: October 14th 2022
% #########################################################################


clear

% datapaths
cd('C:\Users\cstolz\Documents\RevEx1_EEG_Export_BIDS\BIDS_EXPORT\code\'); % select your directory

currdir = pwd;

datapath = strcat(currdir,'\Data');
savepath = strcat(currdir,'\Saves');

cd(datapath);

% load action, feedback, and time interval arrays
load('RevEx1_behavioralData.mat');

% define upper constraints
upper_beta = 10;
upper_alpha = 1;
upper_decay = 1;
upper_k = 1;

REW_fbType = 1; % reward gain (increase positive Q value)
PUN_fbType = 2; % punishment avoidance (decrease negative Q value)

starting_points = 10; % number of different starting points of parameter estimates

rng('default');



for model = 1:5 % models

% random start points
for par = 1:starting_points

   REW_alpharand(par) = rand;
   PUN_alpharand(par) = rand;

   REW_betarand(par) = upper_beta*rand;
   
   PUN_betarand(par) = upper_beta*rand;  
   
   REW_decayrand(par,1) = rand;  
   PUN_decayrand(par,1) = rand;
   
   REW_decayrand(par,2) = rand;  
   PUN_decayrand(par,2) = rand;   
   
   REW_krand(par) = rand;
   PUN_krand(par) = rand;
   
   REW_DTrand(par) = 1;
   PUN_DTrand(par) = 1;   
   
end

    
    for subj = 1:length(REW_action)
        cd(currdir);
        
        options = optimset('Algorithm', 'interior-point', 'Display', 'iter-detailed', 'MaxIter', 10000); % options, increased interations until convergence to 10000
        
        % delete missings/too slow responses (too slow actions = 0)
        REW_action{subj}(REW_action{subj} == 0) = [];
        PUN_action{subj}(PUN_action{subj} == 0) = [];
        
        REW_feedback{subj}(REW_action{subj} == 0) = [];
        PUN_feedback{subj}(PUN_action{subj} == 0) = [];

        REW_LL(subj,model) = 10000;
        PUN_LL(subj,model) = 10000;
   
        
        for rv = 1:10 % 10 different starting points
           disp(strcat('model',num2str(model),{' subject: '},num2str(subj),{' sp: '},num2str(rv),'/10'));

           % ##############################################################
           % execute fmincon using the Q learning function for the
           % reward task, minimizing negative LLE   
           [model_parameters, REW_fminLL] = fmincon(@(x) Q_learning(x,model,REW_action{subj},REW_feedback{subj},REW_fbType,REW_responseFB_interval{subj}),[REW_betarand(rv) REW_alpharand(rv) REW_decayrand(rv) REW_krand(rv) REW_DTrand(rv)],[],[],[],[],[0 0 0 0 0 0 0 1],[upper_beta upper_alpha upper_decay upper_k Inf],[],options);
           % ##############################################################

            if REW_fminLL < REW_LL(subj,model) % update smallest neg LLE
                REW_LL(subj,model) = REW_fminLL;

                if model == 1 % M1: Vanilla
                    REW_alpha(subj) = model_parameters(2);
                    REW_beta_SM(subj) = model_parameters(1);
                    
                elseif model == 2 % M2: Dual updating
                    REW_alpha(subj) = model_parameters(2);
                    REW_beta_SM(subj) = model_parameters(1);                      
                    REW_k_DU(subj) =  model_parameters(4);   

                elseif model == 3 % M3: Time
                    REW_alpha(subj) = model_parameters(2);
                    REW_beta_SM(subj) = model_parameters(1);  
                    REW_decayTime(subj) = model_parameters(5);

                elseif model == 4 % M4: Decay non-chosen positive feedback
                    REW_alpha(subj) = model_parameters(2);
                    REW_beta_SM(subj) = model_parameters(1);  
                    REW_decay(subj) = model_parameters(3);  
                    
                elseif model == 5 % M5: Decay non-chosen all trials
                    REW_alpha(subj) = model_parameters(2);
                    REW_beta_SM(subj) = model_parameters(1);  
                    REW_decay(subj) = model_parameters(3);  
                end;

            end;
            
           % ##############################################################
           % execute fmincon using the Q learning function for the
           % punishment task, minimizing negative LLE            
           [model_parameters, PUN_fminLL] = fmincon(@(x) Q_learning(x,model,PUN_action{subj},PUN_feedback{subj},PUN_fbType,PUN_responseFB_interval{subj}),[PUN_betarand(rv) PUN_alpharand(rv) PUN_decayrand(rv) PUN_krand(rv) PUN_DTrand(rv)],[],[],[],[],[0 0 0 0 0 0 0 1],[upper_beta upper_alpha upper_decay upper_k Inf],[],options);
           % ##############################################################

            if PUN_fminLL < PUN_LL(subj,model) % update smallest neg LLE
                PUN_LL(subj,model) = PUN_fminLL;

                if model == 1 % M1: Vanilla
                    PUN_alpha(subj) = model_parameters(2);
                    PUN_beta_SM(subj) = model_parameters(1);
                    
                elseif model == 2 % M2: Dual updating
                    PUN_alpha(subj) = model_parameters(2);
                    PUN_beta_SM(subj) = model_parameters(1);                      
                    PUN_k_DU(subj) =  model_parameters(4);   

                elseif model == 3 % M3: Time
                    PUN_alpha(subj) = model_parameters(2);
                    PUN_beta_SM(subj) = model_parameters(1);  
                    PUN_decayTime(subj) = model_parameters(5);

                elseif model == 4 % M4: Decay non-chosen positive feedback
                    PUN_alpha(subj) = model_parameters(2);
                    PUN_beta_SM(subj) = model_parameters(1);  
                    PUN_decay(subj) = model_parameters(3);  
                    
                elseif model == 5 % M5: Decay non-chosen all trials
                    PUN_alpha(subj) = model_parameters(2);
                    PUN_beta_SM(subj) = model_parameters(1);  
                    PUN_decay(subj) = model_parameters(3);  
                end;       

        end

        end

    end
     
    cd(savepath);

    if model == 1
       save('RevEx1_fmcon_M1_vanilla.mat','REW_alpha','PUN_alpha','REW_beta_SM','PUN_beta_SM','REW_LL','PUN_LL');
    elseif model == 2
       save('RevEx1_fmcon_M2_Kdupdate.mat','REW_alpha','PUN_alpha','REW_beta_SM','PUN_beta_SM','REW_LL','PUN_LL','REW_k_DU','PUN_k_DU');
    elseif model == 3
       save('RevEx1_fmcon_M3_timeDecay.mat','REW_alpha','PUN_alpha','REW_beta_SM','PUN_beta_SM','REW_LL','PUN_LL','REW_decayTime','PUN_decayTime');  
    elseif model == 4
       save('RevEx1_fmcon_M4_decay.mat','REW_alpha','PUN_alpha','REW_beta_SM','PUN_beta_SM','REW_LL','PUN_LL','REW_decay','PUN_decay');
    elseif model == 5
       save('RevEx1_fmcon_M5_decayAll.mat','REW_alpha','PUN_alpha','REW_beta_SM','PUN_beta_SM','REW_LL','PUN_LL','REW_decay','PUN_decay');   
    end;
    


end
disp('fin');