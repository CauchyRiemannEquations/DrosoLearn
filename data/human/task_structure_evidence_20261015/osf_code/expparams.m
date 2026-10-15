%----------------------------------------------------------------------------
%           Experimental Parameters
%           DO NOT MODIFY between subjects within one experiment
%----------------------------------------------------------------------------

% payout settings 
maxpayout=10.00;           % maximal payout in Euro 
minpayout=0.00; 	     	% minimal payout in Euro 
Z.cent_per_point=0.01; 
load('states_and_rewards.mat'); %get length of exp


%number trials instructions and main experiment  
if doinstr==1;   
    Z.Ntrials       = 40;   % number of learning trials in training (no null events)
else
    Z.Ntrials       = 1000;  % number of trials (with null events)
end



% timing
Z.max_choice_time             =  3;             % maximal time to respond sec
if doeeg                                          
   Z.display_choice           =  1;             % duration to highlight choice with frame without feedback
   Z.display_fix_post_choice  =  1.5;             % ISI for EEG between logged choice and feedback presentation
else
   Z.display_choice           =  0;
   Z.display_fix_post_choice  =  0;
end
Z.display_fb                  =  1;             % duration to display feedback
Z.min_display_fix_cross       =  1;               % + jittered ITI for fMRI; mean 2.5s + 20% null events
pres_ITI=0.3;

% keys 
instrbackward           = 'LeftArrow'; 			% left key for changing instruction page 
instrforward            = 'RightArrow';			% right key for changing instruction page
stopsession = 'Escape'; % key for stoping the experiment
if ~doscanner && ~doeeg
   keyleft     = 'f';          % left key 
   keyright    = 'j';          % right key
   helpbutton  = 'space';      % space key as help button
elseif doscanner == 1
   keyleft     = '2'; % left key, responsbox
   keyright    = '1'; % right key, responsbox 
   trigger     = '5'; % trigger send from the scanner, equals 5 on numpad (BCAN only, MPI Parallelport)
elseif doeeg == 1
   keyleft     = '2'; % left key, responsbox
   keyright    = '1'; % right key, responsbox 
end

% ITIs for scanning
% load predefined variable 'ITI' containing exponentialy distributed
% values: 
if     doscanner
       load('iti_w_null.mat'); 
elseif doeeg
       load('iti_wo_null.mat'); 
       ITI = ITI_wo_null(randperm(160)); 
end

deviceIndex = [];
keysOfInterest = zeros(1,256);
KbQueueCreate(deviceIndex, keysOfInterest);
