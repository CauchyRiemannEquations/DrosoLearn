%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Main shell script 'operant volatility' %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all;  close all;

%config_io;  % initialize parallel port
%outp(888,0);
modifyme;	% set the subject-specific experimental parameters
expparams;  % set parameters that are not specific to subjects 
preps;      % preparations: set up stimulus sequences, left/right etc. 

try 	% this is important: if there's an error, psychtoolbox will crash graciously
		% and move on to the 'catch' block at the end, where all screens etc are
		% closed. 

    if ~debug 
        HideCursor; 
    end
    % set up the psychtoolbox screen and layout parameters 
    % this includes things like positioning of stimuli and loading the
    % stimuli into psychtoolbox 
    setup;
    
    if doscanner || doeeg 
        serial_port = 1;
        config_serial(serial_port, 19200,0,0,8);
        start_cogent;
        if      doscanner
                if doeda
                   isLPT       = 1; %send markers to parallel port for EEG     
                   tpulseLPT   = 25; 
                end
                initialwait;     %if yes wait for MR trigger
        elseif  doeeg
                isLPT       = 1; %send markers to parallel port for EEG     
                tpulseLPT   = 25; 
                initialwait;
        end
    end
    
    %%set states and rewards %%%
    clear S R;
    if doinstr 
        instr_opvol;
        S = [ones(1,20) 2*ones(1, 20)];
        R(:,1) = [Shuffle([ones(1,16), -1, -1, -1, -1]) -Shuffle([ones(1,16), -1, -1, -1, -1])];
        R(:,2) = R(:,1)*(-1); 
    elseif doinstr==0
        switch blockorder
            case 0
                S=regular.S;
                R=regular.R;
            case 1
                S=irregular.S;
                R=irregular.R;
        end
    end

    T.exp_start=GetSecs;
    
    keysOfInterest = zeros(1,256);
    keysOfInterest(KbName(keyleft)) = 1;
    keysOfInterest(KbName(keyright)) = 1;
    keysOfInterest(KbName(helpbutton)) = 1;
    keysOfInterest(KbName(stopsession)) = 1;
    KbQueueCreate(deviceIndex, keysOfInterest);
 
    for nt = 1:Z.Ntrials
        opvoltrial
    end
        
    
    T.exp_end = GetSecs; 
  
    if doscanner==1; T.baseline_end = GetSecs; WaitSecs(10-monitorFlipInterval); end % 10 sec baseline at the end
    if doinstr; Screen('TextSize',wd,txtsize); instr_opvol_posttraining; end
    if payment && doinstr == 0; instr_payment; end
    if ~doinstr; WaitSecs(3-monitorFlipInterval); end
    if doscanner
	   stop_cogent;
    end

    %---------------------------------------------------------------------------
	fprintf('saving all in three separate files\n');

	if dosave; eval(['save data' filesep namestring_long '_WS.mat']);end
	savepath = ['data' filesep namestring_long];
    if dosave; eval('save(savepath, ''A'',''C'',''R'',''RT'',''S'',''T'')');end
    savepath = ['data' filesep namestring_short];
    if dosave; eval('save(savepath, ''A'',''C'',''R'',''RT'',''S'',''T'')');end 
    ShowCursor; % show the mouse cursor again 
	Screen('CloseAll');
    fprintf('.........done\n'); 
    
catch % execute this if there's an error, or if we've pressed the escape key
      Screen('CloseAll'); % close psychtoolbox, return screen control to OSX
      ShowCursor;
      if     aborted==0;	 % if there was an error
		     fprintf(' ******************************\n')
		     fprintf(' **** Something went WRONG ****\n')
		     fprintf(' ******************************\n')
		     if dosave; eval(['save data' filesep namestring_long '.crashed.mat;']);end
      elseif aborted==1; % if we've abored by pressing the escape key
		     fprintf('                               \n')
		     fprintf(' ******************************\n')
		     fprintf(' **** Experiment aborted ******\n')
		     fprintf(' ******************************\n')
		     if dosave; eval(['save data' filesep namestring_long  '.aborted.mat;']);end
      end
      if dosave;  
          eval(['save data' filesep namestring_long '-' date '.mat;']); 
          fclose('all');
      end
      rethrow(lasterror);
end