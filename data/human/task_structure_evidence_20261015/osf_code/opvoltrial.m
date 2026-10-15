fprintf(['............. Trial number ' num2str(nt) '\n'])
Screen('TextSize', wd, txtsize);

% Get images for instructions or main experiment 
if doinstr == 1
    dreieck_allg = cue3;
    viereck_allg = cue4;
elseif doinstr == 0    
    dreieck_allg = cue1;
    viereck_allg = cue2;
end


Screen('DrawTexture',wd,dreieck_allg,[],box(random_lr(nt),:));
Screen('DrawTexture',wd,viereck_allg, [],box(3-random_lr(nt),:));

% Draw hint button
draw_hints;

% Show stimulus on screen at next possible display refresh cycle,
% and record stimulus onset time in 'startrt':
[T.trial_onset(nt)] = Screen('Flip', wd);
KbQueueStart(deviceIndex);

if doeda | doeeg 
    outp(888,1); wait(tpulseLPT);  outp(888,0);
end

% Get choices 
% while loop to show stimulus until subjects response or until "duration"
% seconds elapsed.
secs = GetSecs;
while (GetSecs - T.trial_onset(nt)) <= Z.max_choice_time-monitorFlipInterval
    [KeyIsDown, firstPress] = KbQueueCheck(deviceIndex);
    press_secs = firstPress(find(firstPress));
    if KeyIsDown
        key = KbName(min(find(firstPress)));
        if strcmp(key, keyleft)
            valid_choice = 1;
            RT(nt) = press_secs - secs;
            redraw = 1;
            a_side(nt) = 1;
        elseif strcmp(key, keyright)            
            valid_choice = 1;
            RT(nt) = press_secs - secs;
            redraw = 1;
            a_side(nt) = 2;
        elseif strcmp(key, helpbutton)            
            valid_choice = 1;
            RT(nt) = press_secs - secs;
            redraw = 1;
            a_side(nt) = 3;
        elseif strcmp(key, 'ESCAPE')
            valid_choice = -1;
            checkabort;
        end
        break
    else
        valid_choice = -1;
        RT(nt) = NaN;
        a_side(nt) = NaN;
        key='no_response';
    end
end
KbQueueStop(deviceIndex);	% Stop delivering events to the queue

% Display feedback
if valid_choice==1 
   if   random_lr(nt)==2; 
       A(nt)=a_side(nt);
   else
       if a_side(nt) == 1
           A(nt) = 2;
       elseif a_side(nt) == 2
           A(nt) = 1;
       elseif a_side(nt) == 3
           A(nt) = a_side(nt);
       end
   end

   % compute consequences 
   if S(nt) == A(nt)      
       C(nt) =  1;
   elseif A(nt) == 3        % here, no consequences happen when helpbutton is chosen 
       C(nt) = NaN;
   elseif  S(nt) ~= A(nt)                 % 0 or 3 (help button) would also fall in this category     
       C(nt) =  0;
   end

   % problem: rewards are saved in R, which consists of only 2 columns
   % when help button is pressed, one would access the 0th or 3rd column of
   % R -> therefore the if statement
   if A(nt) == 3
       Rew(nt) = 0;
   else
       Rew(nt)=R(nt,A(nt));
   end
   
   %........................ Draw boxes and frame arround chosen square
   % Display choice
   
   if a_side(nt) == 3   % helpbutton pressed
       Screen('DrawTexture',wd,squareframe,[],boxl(S(nt),:));       % correct answer(S(nt)) is framed
       centBar = CenterRectOnPoint([0 0 420 120], wdw/2, ypos_expl - 50);
       Screen('FillRect', wd, bluerect, centBar);
   else    
       Screen('DrawTexture',wd,squareframe,[],boxl(a_side(nt),:));
   end
   % Draw hint button
   draw_hints;
   
   Screen('DrawTexture',wd,dreieck_allg,[],box(random_lr(nt),:));
   Screen('DrawTexture',wd,viereck_allg,  [],box(3-random_lr(nt),:));
   Screen('Flip', wd);

   
   if doeda | doeeg 
      if     a_side(nt) == 1
             marker_lr = 2;
      elseif a_side(nt) == 2
             marker_lr = 3; 
      end
      outp(888,marker_lr); wait(tpulseLPT);  outp(888,0);
   end
   WaitSecs(Z.display_choice-monitorFlipInterval);
   
   if doeeg
      Screen('TextSize',wd,txt_fix);
      DrawFormattedText(wd,'+','center','center',txtcolor);
      T.post_choice_fix(nt) = Screen('Flip', wd);
      WaitSecs(Z.display_fix_post_choice-monitorFlipInterval); 
   end

   % Display feedback... 
   if     Rew(nt)== 1
          Screen('DrawTexture',wd,smiley,[],box_center);
          if doeda | doeeg 
             fb_marker = 4;
          end
   elseif Rew(nt)== -1
          Screen('DrawTexture',wd,frowny,[],box_center);
          if doeda | doeeg 
             fb_marker = 5; 
          end
   end
   % ... and write outcome explicitly, too 
   if     Rew(nt)== 1
          text1 = txt_win; 
          text2 = cent_win; 
          ocol  = col_gr;
          ypos1= box_center(2)-20;

   elseif Rew(nt)== -1
          text1 = txt_loss; 
          text2 = cent_loss; 
          ocol  = col_re;
          ypos1= box_center(2)-20;  

   elseif Rew(nt) == 0
       text1 = txt_help;
       ocol = col_wh;
       ypos1 = box_center(2);
   end
   
   % Display feedback and choice 
   if a_side(nt) == 3
       Screen('DrawTexture',wd,squareframe,[],boxl(S(nt),:));        % help button: correct solution (S(nt)) is shown
       centBar = CenterRectOnPoint([0 0 420 120], wdw/2, ypos_expl - 50);
       Screen('FillRect', wd, bluerect, centBar);
   else
       Screen('DrawTexture',wd,squareframe,[],boxl(a_side(nt),:));   % only choosing left/right needs a text2 option
       [wt]=Screen(wd,'TextBounds',text2)                            % when help is pressed, text2 is unnecessary
       ypos2= box_center(4)+1.5*wt(4);
       DrawFormattedText(wd,text2,'center',ypos2,ocol);   
   end
   
   % Draw hint button
   draw_hints;
   Screen('DrawTexture',wd,dreieck_allg,[],box(  random_lr(nt),:));
   Screen('DrawTexture',wd,viereck_allg,  [],box(3-random_lr(nt),:));
   DrawFormattedText(wd,text1,'center',ypos1,ocol);     
   
   T.onset_fb(nt) = Screen('Flip', wd);
   
   if doeda | doeeg 
       outp(888,fb_marker); wait(tpulseLPT);  outp(888,0);
   end
   WaitSecs(Z.display_fb-monitorFlipInterval);
   
elseif valid_choice==-1
% Display feedback when response was too slow
       T.onset_choice(nt) = NaN; 
       if doeeg
          Screen('TextSize',wd,txt_fix);
          DrawFormattedText(wd,'+','center','center',txtcolor);
          T.post_choice_fix(nt) = Screen('Flip', wd);
          WaitSecs(Z.display_fix_post_choice-monitorFlipInterval); 
       end
       fb_marker = 6;
       if doeeg
           DrawFormattedText(wd,text_fb_too_slow,'center','center',red);
       else
           draw_hints;
           DrawFormattedText(wd,text_fb_too_slow,'center',ypos_fb,red);
           Screen('DrawTexture',wd,dreieck_allg,[],box(  random_lr(nt),:));
           Screen('DrawTexture',wd,viereck_allg, [],box(3-random_lr(nt),:));
       end
       Screen('Flip', wd);
       if doeda | doeeg 
          outp(888,fb_marker); wait(tpulseLPT);  outp(888,0);
       end
       WaitSecs(Z.display_fb-monitorFlipInterval);
  
       A(nt)   = NaN;
       R(nt)   = NaN;
       C(nt)   = NaN;
end 
       

% Wait ITI and show fixationcross
Screen('TextSize',wd,txt_fix);
DrawFormattedText(wd,'+','center','center',txtcolor);
Screen('Flip', wd);

WaitSecs(pres_ITI-monitorFlipInterval);