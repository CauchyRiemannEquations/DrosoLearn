% #########################################################################
% Single-trial regression analysis
%
%   prediction error -> Frontal Theta power
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
% Version 1: October 15th 2022
% #########################################################################


%% ########################################################################
% Single-trial analysis computationally-derived prediction errors on frontal theta power
% #########################################################################
% This script runs a single-trial analysis of single-trial computationally-
% derived prediction errors on theta for signed prediction errors. Btx =
% 1 better than expected (PE+) and wtx = 0 worse than expected (PE-).
%
% EEG data format: Bins_REW{subject}(channel, data point, frequency, trial)
% I run the power normalization in this script so the final analyses
% were conducted using the variables Bins_REW_c and Bins_PUN_c.
% 
% Feel free to adapt the cide to your data; you can also run it for
% event-related potential data by removing the "frequency" index and the
% freq loop. You % can also easily run analyses of EEG activity on
% subsequent behavioral adjustment (e.g. reaction time) by defining
% Bins_REW_c as x1 and the % behavioural data as y variable. Just keep in
% mind that if you want to % analyze the relationship between the current 
% and subsequent trial that you need to arrange the arrays accordingly:
% (EEG trial n -> behavior trial n+1). You may % want to delete the last
% index of x1 and fb, i.e. x1(end) = []; & fb(end) = []; % so the arrays
% have the same length (here it would be 279). I also added an example
% for a multiple regression analysis below.

clear

for model = 1:5 % define the model of interest here

cd('C:\Users\cstolz\Documents\RevEx1_EEG_Export_BIDS\BIDS_EXPORT\code\'); % select your directory
currdir = pwd;

datapath = strcat(currdir,'\Data');
savepath = strcat(currdir,'\Saves');
eegpath = strcat(currdir,'\Data\eeg');

cd(datapath)
load('RevEx1_behavioralData.mat');
load('RevEx1_BadEEGTrials.mat');

cd(savepath);
loadname = strcat('RevEx1_PR_M',num2str(model),'.mat');
load(loadname);

% #########################################################################
% set time-frequency range of interest; depends on your data structure
freqrange = 7:12; % theta 4-8    
timerange = 283:411; % 250 bis 500 ms
% #########################################################################


% load EEG data, here 6 files for frontal EEG channels due to memory.
% I extracted the time-frequency of interest single-trial information,
% otherwise it would cause memory issues.
cd(eegpath)
%%%%%%%%%%%%%%%%%%%%%%%
for chancounter = 1:6
%%%%%%%%%%%%%%%%%%%%%%%
    
    loadname = strcat('RevEx1_REW_FrontalChannel',num2str(chancounter),'.mat');
    load(loadname);
    
    loadname = strcat('RevEx1_PUN_FrontalChannel',num2str(chancounter),'.mat');
    load(loadname);    

    Bins_REW(kickers) = [];
    Bins_PUN(kickers) = [];
    
disp('loading finished');


% #########################################################################
% normalizing time-frequency data to dB
% #########################################################################
 freq_a = 1;
 
 freq_b = 25;
 
    bslcorr_start = 1; % -300

    bslcorr_end = 53; % -200

    for subj = 1:length(Bins_PUN)
            for channel = 1:length(Bins_PUN{subj}(:,1,1,1))
              for trialindex = 1:length(Bins_PUN{subj}(1,1,1,:));  
                  for freq = 1:length(Bins_PUN{subj}(channel,1,:,trialindex))
                      
                    % REW dB
                    Bins_REW{subj}(channel,:,freq,trialindex) = 10*log10(bslcorrWAMat_div(Bins_REW{subj}(channel,:,freq,trialindex),bslcorr_start:bslcorr_end));

                    % PUN dB
                    Bins_PUN{subj}(channel,:,freq,trialindex) = 10*log10(bslcorrWAMat_div(Bins_PUN{subj}(channel,:,freq,trialindex),bslcorr_start:bslcorr_end));
                    
                  end;
              end;

            end;  
    end; 
% #########################################################################

% #########################################################################
% extract time-frequency data points of interest (here 4-8 Hz, 250-500 ms)
% #########################################################################
        for subj = 1:length(Bins_PUN)
           % REW avg
           for trial = 1:length(Bins_REW{subj}(1,1,1,:))
                %REW
                Bins_REW_c{subj}(chancounter,:,:,trial) = Bins_REW{subj}(1,timerange,freqrange,trial);
           end
           
           % PUN avg
           for trial = 1:length(Bins_PUN{subj}(1,1,1,:))
                %PUN
                Bins_PUN_c{subj}(chancounter,:,:,trial) = Bins_PUN{subj}(1,timerange,freqrange,trial);
           end
        end
   
end
% #########################################################################
    

disp('Calculate regressions: PR --> EEG');
disp('---------------------------------');
    
    for btx_wtx = 0:1 % worse than expected = 0, better than expected = 1
    
        for subject = 1:length(Bins_REW_c) % subject loop
    
             disp(cell2mat(strcat('VP:',{' ' },num2str(subject),'/',num2str(length(Bins_REW_c)))));

             for freqer = 1:length(freqrange) % frequency loop
        
                for channel = 1:length(Bins_REW_c{subject}(:,1,1,1)) % channel loop, here 1
        
                        for bin = 1:length(Bins_REW_c{subject}(1,:,1,1)) % bin/data point loop
                            % ############################################
                            % Reward task
                            % ############################################ 

                            % y, EEG data
                            y = Bins_REW_c{subject}(channel,bin,freqer,:);
                            y = y(:);
        
                            % x1, prediction errors
                            x1 =  abs(PR_REW_gen{subject}); % |PR|
                            x1 = x1(:);
                            
                            % feedback array
                            fb = REW_feedback{subject};
        
                            % erase bad intervals
                            y(bad_trials_REW{subject}) = [];
                            x1(bad_trials_REW{subject}) = [];
                            fb(bad_trials_REW{subject}) = [];
                            
                            % save positive / negative feedback trials
                            y = y(fb==btx_wtx);
                            x1 = x1(fb==btx_wtx);
        

                            % single trial regression x1 -> y
                            X = [ones(size(x1)) zscore(x1)];
                            B = regress(zscore(y),X);
                            slope_PR_EEG_REW{subject}(channel,freqer,bin) = B(2); % slope prediction error -> EEG
                            intercept_PR_EEG_REW{subject}(channel,freqer,bin) = B(1); % intercept prediction error -> EEG 

                    
                            % ############################################ 
                            % example multiple regression
                            % ############################################ 

                            % if you want to calculate a multiple
                            % regression analysis, you can add x2, x3, x4,...
                            % and interaction terms to the X variable.
                            
                                %{
    
                                X = [ones(size(x1)) zscore(x1) zscore(x2) zscore(x1).*zscore(x2)];
                                B = regress(zscore(y1),X);
                                slope_x1{heino}(channel,freqer,bin) = B(2);
                                slope_x2{heino}(channel,freqer,bin) = B(3);
                                slope_x1x2{heino}(channel,freqer,bin) = B(4);
                                intercept{subject}(channel,freqer,bin) = B(1);
    
                                %}
                        end
                         
                        for bin = 1:length(Bins_PUN_c{subject}(1,:,1,1))
                                    % ############################################
                            % Punishment task
                            % ############################################ 

                            % y, EEG data
                            y = Bins_PUN_c{subject}(channel,bin,freqer,:);
                            y = y(:);
        
                            % x1, prediction errors
                            x1 =  abs(PR_PUN_gen{subject}); % |PR|
                            x1 = x1(:);
                            
                            % feedback array
                            fb = PUN_feedback{subject};
        
                            % erase bad intervals
                            y(bad_trials_PUN{subject}) = [];
                            x1(bad_trials_PUN{subject}) = [];
                            fb(bad_trials_PUN{subject}) = [];
                            
                            % save positive / negative feedback trials
                            y = y(fb==btx_wtx);
                            x1 = x1(fb==btx_wtx);
        

                            % single trial regression x1 -> y
                            X = [ones(size(x1)) zscore(x1)];
                            B = regress(zscore(y),X);
                            slope_PR_EEG_PUN{subject}(channel,freqer,bin) = B(2); % slope prediction error -> EEG
                            intercept_PR_EEG_PUN{subject}(channel,freqer,bin) = B(1); % intercept prediction error -> EEG  
        
                        end;
        
                end;
        
             end;
         
         % average slopes/intecepts in the time-frequency range of interest
         % for every participant
         slope_PR_EEG_REW_avg(subject) = mean(mean(mean(slope_PR_EEG_REW{subject},3)));
         intercept_PR_EEG_REW_avg(subject) = mean(mean(mean(intercept_PR_EEG_REW{subject},3)));     
         
         slope_PR_EEG_PUN_avg(subject) = mean(mean(mean(slope_PR_EEG_PUN{subject},3)));
         intercept_PR_EEG_PUN_avg(subject) = mean(mean(mean(intercept_PR_EEG_PUN{subject},3)));
            
         clear pk
         pk = slope_PR_EEG_REW{subject}(channel,freqer,1:65);
         slope_PR_EEG_REW_avg_early(subject) = mean(mean(mean(pk,3)));
         pk = slope_PR_EEG_REW{subject}(channel,freqer,66:end);
         slope_PR_EEG_REW_avg_late(subject) = mean(mean(mean(pk,3)));
    
    
         pk = slope_PR_EEG_PUN{subject}(channel,freqer,1:65);
         slope_PR_EEG_PUN_avg_early(subject) = mean(mean(mean(pk,3)));
         pk = slope_PR_EEG_PUN{subject}(channel,freqer,66:end);
         slope_PR_EEG_PUN_avg_late(subject) = mean(mean(mean(pk,3)));
         clear pk
    
        end;
        
        % save better than expected and worse than expected slopes
        cd(savepath);
        if btx_wtx == 0
           save(strcat('RevEx1_regression_WTXpr_Theta_Front_M',num2str(model),'.mat'),'slope_PR_EEG_PUN','slope_PR_EEG_REW','intercept_PR_EEG_PUN','intercept_PR_EEG_REW','slope_PR_EEG_REW_avg','intercept_PR_EEG_REW_avg','slope_PR_EEG_PUN_avg','intercept_PR_EEG_PUN_avg','slope_PR_EEG_REW_avg_early','slope_PR_EEG_REW_avg_late','slope_PR_EEG_PUN_avg_early','slope_PR_EEG_PUN_avg_late');
        elseif btx_wtx == 1
           save(strcat('RevEx1_regression_BTXpr_Theta_Front_M',num2str(model),'.mat'),'slope_PR_EEG_PUN','slope_PR_EEG_REW','intercept_PR_EEG_PUN','intercept_PR_EEG_REW','slope_PR_EEG_REW_avg','intercept_PR_EEG_REW_avg','slope_PR_EEG_PUN_avg','intercept_PR_EEG_PUN_avg','slope_PR_EEG_REW_avg_early','slope_PR_EEG_REW_avg_late','slope_PR_EEG_PUN_avg_early','slope_PR_EEG_PUN_avg_late');
        end %save
    
    end %btxwtx

end %model



%% ########################################################################
% Single-trial analysis self-reported prediction errors on frontal theta power
% #########################################################################
% This script runs a single-trial analysis of single-trial self-reported
% prediction errors on theta for signed prediction errors. Btx =
% 1 better than expected (PE+) and wtx = 0 worse than expected (PE-).
%

clear

cd('C:\Users\cstolz\Documents\RevEx1_EEG_Export_BIDS\BIDS_EXPORT\code\'); % select your directory
currdir = pwd;

datapath = strcat(currdir,'\Data');
savepath = strcat(currdir,'\Saves');
eegpath = strcat(currdir,'\Data\eeg');

cd(datapath)
load('RevEx1_behavioralData.mat');
load('RevEx1_BadEEGTrials.mat');

cd(savepath);
loadname = strcat('RevEx1_PR_M',num2str(model),'.mat');
load(loadname);

% #########################################################################
% set time-frequency range of interest; depends on your data structure
freqrange = 7:12; % theta 4-8    
timerange = 283:411; % 250 bis 500 ms
% #########################################################################


% #########################################################################
% compute self-reported prediction errors based on single-trial expectation
% ratings

% transform expectation ratings to 0, 1/3, 2/3, 1
for subj = 1:length(REW_expect)
   REW_expect{subj}( REW_expect{subj} == 3) = 0; 
   REW_expect{subj}( REW_expect{subj} == 4) = 1/3; 
   REW_expect{subj}( REW_expect{subj} == 5) = 2/3; 
   REW_expect{subj}( REW_expect{subj} == 6) = 1; 
   
   PUN_expect{subj}( PUN_expect{subj} == 3) = 0; 
   PUN_expect{subj}( PUN_expect{subj} == 4) = 1/3; 
   PUN_expect{subj}( PUN_expect{subj} == 5) = 2/3; 
   PUN_expect{subj}( PUN_expect{subj} == 6) = 1; 
end;

% self report PR
for subj = 1:length(REW_expect)
    for trial = 1:length(REW_expect{subj})
        REW_srPR{subj}(trial) = REW_feedback{subj}(trial) - REW_expect{subj}(trial);
        PUN_srPR{subj}(trial) = PUN_feedback{subj}(trial) - PUN_expect{subj}(trial);
    end
end
% #########################################################################


% load EEG data, here 6 files for frontal EEG channels due to memory.
% I extracted the time-frequency of interest single-trial information,
% otherwise it would cause memory issues.
cd(eegpath)
%%%%%%%%%%%%%%%%%%%%%%%
for chancounter = 1:6
%%%%%%%%%%%%%%%%%%%%%%%
    
    loadname = strcat('RevEx1_REW_FrontalChannel',num2str(chancounter),'.mat');
    load(loadname);
    
    loadname = strcat('RevEx1_PUN_FrontalChannel',num2str(chancounter),'.mat');
    load(loadname);    

    Bins_REW(kickers) = [];
    Bins_PUN(kickers) = [];
    
    disp('loading finished');


% #########################################################################
% normalizing time-frequency data to dB
% #########################################################################
    bslcorr_start = 1; % -300
    bslcorr_end = 53; % -200

    for subj = 1:length(Bins_PUN)
            for channel = 1:length(Bins_PUN{subj}(:,1,1,1))
              for trialindex = 1:length(Bins_PUN{subj}(1,1,1,:));  
                  for freq = 1:length(Bins_PUN{subj}(channel,1,:,trialindex))
                      
                    % REW dB
                    Bins_REW{subj}(channel,:,freq,trialindex) = 10*log10(bslcorrWAMat_div(Bins_REW{subj}(channel,:,freq,trialindex),bslcorr_start:bslcorr_end));

                    % PUN dB
                    Bins_PUN{subj}(channel,:,freq,trialindex) = 10*log10(bslcorrWAMat_div(Bins_PUN{subj}(channel,:,freq,trialindex),bslcorr_start:bslcorr_end));
                    
                  end;
              end;

            end;  
    end; 
% #########################################################################

% #########################################################################
% extract time-frequency data points of interest (here 4-8 Hz, 250-500 ms)
% #########################################################################
    for subj = 1:length(Bins_PUN)
       % REW avg
       for trial = 1:length(Bins_REW{subj}(1,1,1,:))
            %REW
            Bins_REW_c{subj}(chancounter,:,:,trial) = Bins_REW{subj}(1,timerange,freqrange,trial);
       end

       % PUN avg
       for trial = 1:length(Bins_PUN{subj}(1,1,1,:))
            %PUN
            Bins_PUN_c{subj}(chancounter,:,:,trial) = Bins_PUN{subj}(1,timerange,freqrange,trial);
       end
    end
end
% #########################################################################
    

disp('Calculate regressions: SR --> EEG');
disp('---------------------------------');
    
for btx_wtx = 0:1 % worse than expected = 0, better than expected = 1

    for subject = 1:length(Bins_REW_c) % subject loop

         disp(cell2mat(strcat('VP:',{' ' },num2str(subject),'/',num2str(length(Bins_REW_c)))));

         for freqer = 1:length(freqrange) % frequency loop

            for channel = 1:length(Bins_REW_c{subject}(:,1,1,1)) % channel loop, here 1

                    for bin = 1:length(Bins_REW_c{subject}(1,:,1,1)) % bin/data point loop
                        % ############################################
                        % Reward task
                        % ############################################ 

                        % y, EEG data
                        y = Bins_REW_c{subject}(channel,bin,freqer,:);
                        y = y(:);

                        % x1, prediction errors
                        x1 =  abs(REW_srPR{subject}); % |PR|
                        x1 = x1(:);

                        % feedback array
                        fb = REW_feedback{subject};

                        % erase bad intervals
                        y(bad_trials_REW{subject}) = [];
                        x1(bad_trials_REW{subject}) = [];
                        fb(bad_trials_REW{subject}) = [];

                        % save positive / negative feedback trials
                        y = y(fb==btx_wtx);
                        x1 = x1(fb==btx_wtx);


                        % single trial regression x1 -> y
                        X = [ones(size(x1)) zscore(x1)];
                        B = regress(zscore(y),X);
                        slope_PR_EEG_REW{subject}(channel,freqer,bin) = B(2); % slope prediction error -> EEG
                        intercept_PR_EEG_REW{subject}(channel,freqer,bin) = B(1); % intercept prediction error -> EEG 


                        % ############################################ 
                        % example multiple regression
                        % ############################################ 

                        % if you want to calculate a multiple
                        % regression analysis, you can add x2, x3, x4,...
                        % and interaction terms to the X variable.

                            %{

                            X = [ones(size(x1)) zscore(x1) zscore(x2) zscore(x1).*zscore(x2)];
                            B = regress(zscore(y1),X);
                            slope_x1{heino}(channel,freqer,bin) = B(2);
                            slope_x2{heino}(channel,freqer,bin) = B(3);
                            slope_x1x2{heino}(channel,freqer,bin) = B(4);
                            intercept{subject}(channel,freqer,bin) = B(1);

                            %}
                    end

                    for bin = 1:length(Bins_PUN_c{subject}(1,:,1,1))
                                % ############################################
                        % Punishment task
                        % ############################################ 

                        % y, EEG data
                        y = Bins_PUN_c{subject}(channel,bin,freqer,:);
                        y = y(:);

                        % x1, prediction errors
                        x1 =  abs(PUN_srPR{subject}); % |PR|
                        x1 = x1(:);

                        % feedback array
                        fb = PUN_feedback{subject};

                        % erase bad intervals
                        y(bad_trials_PUN{subject}) = [];
                        x1(bad_trials_PUN{subject}) = [];
                        fb(bad_trials_PUN{subject}) = [];

                        % save positive / negative feedback trials
                        y = y(fb==btx_wtx);
                        x1 = x1(fb==btx_wtx);


                        % single trial regression x1 -> y
                        X = [ones(size(x1)) zscore(x1)];
                        B = regress(zscore(y),X);
                        slope_PR_EEG_PUN{subject}(channel,freqer,bin) = B(2); % slope prediction error -> EEG
                        intercept_PR_EEG_PUN{subject}(channel,freqer,bin) = B(1); % intercept prediction error -> EEG  

                    end;

            end;

         end;

     % average slopes/intecepts in the time-frequency range of interest
     % for every participant
     slope_PR_EEG_REW_avg(subject) = mean(mean(mean(slope_PR_EEG_REW{subject},3)));
     intercept_PR_EEG_REW_avg(subject) = mean(mean(mean(intercept_PR_EEG_REW{subject},3)));     

     slope_PR_EEG_PUN_avg(subject) = mean(mean(mean(slope_PR_EEG_PUN{subject},3)));
     intercept_PR_EEG_PUN_avg(subject) = mean(mean(mean(intercept_PR_EEG_PUN{subject},3)));

     clear pk
     pk = slope_PR_EEG_REW{subject}(channel,freqer,1:65);
     slope_PR_EEG_REW_avg_early(subject) = mean(mean(mean(pk,3)));
     pk = slope_PR_EEG_REW{subject}(channel,freqer,66:end);
     slope_PR_EEG_REW_avg_late(subject) = mean(mean(mean(pk,3)));


     pk = slope_PR_EEG_PUN{subject}(channel,freqer,1:65);
     slope_PR_EEG_PUN_avg_early(subject) = mean(mean(mean(pk,3)));
     pk = slope_PR_EEG_PUN{subject}(channel,freqer,66:end);
     slope_PR_EEG_PUN_avg_late(subject) = mean(mean(mean(pk,3)));
     clear pk

    end;

    % save better than expected and worse than expected slopes
    cd(savepath);
    if btx_wtx == 0
       save(strcat('RevEx1_regression_WTXsr_Theta_Front.mat'),'slope_PR_EEG_PUN','slope_PR_EEG_REW','intercept_PR_EEG_PUN','intercept_PR_EEG_REW','slope_PR_EEG_REW_avg','intercept_PR_EEG_REW_avg','slope_PR_EEG_PUN_avg','intercept_PR_EEG_PUN_avg','slope_PR_EEG_REW_avg_early','slope_PR_EEG_REW_avg_late','slope_PR_EEG_PUN_avg_early','slope_PR_EEG_PUN_avg_late');
    elseif btx_wtx == 1
       save(strcat('RevEx1_regression_BTXsr_Theta_Front.mat'),'slope_PR_EEG_PUN','slope_PR_EEG_REW','intercept_PR_EEG_PUN','intercept_PR_EEG_REW','slope_PR_EEG_REW_avg','intercept_PR_EEG_REW_avg','slope_PR_EEG_PUN_avg','intercept_PR_EEG_PUN_avg','slope_PR_EEG_REW_avg_early','slope_PR_EEG_REW_avg_late','slope_PR_EEG_PUN_avg_early','slope_PR_EEG_PUN_avg_late');
    end %save

end %btxwtx

%% load files 