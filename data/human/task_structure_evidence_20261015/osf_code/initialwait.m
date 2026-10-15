if ~doscanner && ~doeeg;     fprintf('............. Outside scanner \n');
   T.time_begin = GetSecs; 

elseif doeeg==1 && ~doinstr; fprintf('............. EEG experiment \n');
                             fprintf('............. Waiting for button press  \n');
       Screen('TextSize',wd,txtlarge);
	   DrawFormattedText(wd,'Bitte Enter zu Starten drücken!',yposm,yposm,txtcolor);
	   Screen('Flip',wd);
	   
       while 1
             [secs, keyCode, deltaSecs] = KbStrokeWait; % wait for button press 
             button = KbName(keyCode); 
             if button == 'Return';
                break
             end
       end
       
elseif doscanner==1; fprintf('............. Inside scanner \n');
                     fprintf('............. Waiting for trigger \n');
       Screen('TextSize',wd,txtlarge);
	   DrawFormattedText(wd,'Gleich geht es los...',yposm,yposm,txtcolor);
	   Screen('Flip',wd);
	   % wait for trigger before starting   
	   wait_for_trigger_matlab75
       Screen('TextSize',wd,txt_fix);
       DrawFormattedText(wd,'+','center','center',txtcolor);
       T.baseline_start = Screen('Flip',wd);
       WaitSecs(5-monitorFlipInterval); % 5 sec baseline befor experiment starts
       Screen('TextSize',wd,txtsize);
end
if ~doinstr
    Screen('TextSize',wd,txt_fix);
    DrawFormattedText(wd,'+','center','center',txtcolor);
    T.baseline_start = Screen('Flip',wd);
    WaitSecs(5-monitorFlipInterval); % 5 sec baseline befor experiment starts
    Screen('TextSize',wd,txtsize);
end