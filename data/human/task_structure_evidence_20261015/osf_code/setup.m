fprintf('............ Setting up the screen   \n');

% open a screen
PsychDefaultSetup(2);
PsychTweak('UseGPUIndex', 0);
screen_number = max(Screen('Screens'));

% colours (in RBG)
                        % black 		= [0 0 0]; 
                        % gray1   	= [70 70 70];
                        % purple		= [148 0 211];
                        % brown		= [205 133 63];
                        % chartreuse 	= [127 255 0];
                        % yellow		= [250 250 50]; 
                        % purple		= [150 0 150];

bgcol 		= [80 0]/255;	% this is just in grayscale (each value separately)
white 		= [200 200 200]/255;
hard_white = WhiteIndex(screen_number);
blackish    = hard_white/20;
red 		= [255 20 20]/255; 
blue 		= [120 0 255]/255;
bluetxt     = [41 94 146]/255;
green       = [0 135 0]/255;
greenrect   = [154 192 182]/255;
yellowrect  = [255 217 102]/255;
bluerect    = [41 94 146]/255;
txtcolor 	= white;
txtsize     = 27; % text size
txtlarge    = 27;  
blw         = .2;                              % width of stimulus as fraction of **xfrac**
blh         = .2;                              % height of stimulus as fraction of **xfrac**
txt_fix     = 50; 



if debug; 
	% Screen('Preference','SkipSyncTests',2); % ONLY do this for quick debugging;
    PsychDebugWindowConfiguration();
end

[wd, windowRect] = PsychImaging('OpenWindow',screen_number,blackish);

% wd=Screen('OpenWindow',0,bgcol(2),[],[],2,[],[],[]);	     % Get Screen. This is always size of the display.

Screen('BlendFunction', wd, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
Screen('TextSize',wd,txtsize);				% Set size of text

% Do dummy calls to GetSecs, WaitSecs, to make sure
% they are loaded and ready when we need them - without delays
% in the wrong moment.
WaitSecs(0.01);
GetSecs;

% AT MPI computers this priority level made KbCheck crash! 
% Set priority for script execution to realtime priority:
%priorityLevel = MaxPriority(wd, ['GetSecs'],['WaitSecs'],['KbCheck'],['KbWait']);
%Priority(priorityLevel);

%---------------------------------------------------------------------------
%                    SCREEN LAYOUT
%---------------------------------------------------------------------------
[wdw, wdh]=Screen('WindowSize', wd);	% Get screen size 

%................... Presentation coordinates 
xfrac=.7; 				% fraction of x width to use 
yfrac=.5; 				% fraction of y height to use 
xl0=xfrac*wdw; 			% width to use in pixels
yl0=yfrac*wdh; 			% height to use in pixels
x0=(1-xfrac)/2*wdw; 	% zero point along width 
y0=(1-yfrac)/2*wdh;		% zero point along height

%.................... The squares 
boxc = x0+round([xl0*1/6  xl0*5/6]);                    % x centres left and right third of fracdisplay
box0 = round([-blw*xl0 -blh*xl0 blw*xl0 blh*xl0]/2);    % boxlines 

% each box
for k=1:2;
	box (k,:) =     box0 + [boxc(k) wdh/2.1 boxc(k) wdh/1.8];	% main boxes 
	boxl(k,:) = 1.2*box0 + [boxc(k) wdh/2.1 boxc(k) wdh/1.8];% slightly larger box 
    %boxs(k,:) = 0.8*box0 + [boxc(k) wdh/2.1 boxc(k) wdh/1.8];%slightly smaller box
end
box_center    = 0.8*box0 + [wdw/2 wdh/2 wdw/2 wdh/2];

squareframe= Screen('MakeTexture',wd,repmat(reshape(blue,[1 1 3]),[5 5 1]));

randind=[1 2 3 4];
eval(['tmp=imread(''imgs' filesep 'Folie' num2str(randind(3)) '.JPG'');'])
cue3=Screen('MakeTexture',wd,tmp);
eval(['tmp=imread(''imgs' filesep 'Folie' num2str(randind(4)) '.JPG'');'])
cue4=Screen('MakeTexture',wd,tmp);
eval(['tmp=imread(''imgs' filesep 'Folie' num2str(randind(1)) '.JPG'');'])
cue1=Screen('MakeTexture',wd,tmp);
eval(['tmp=imread(''imgs' filesep 'Folie' num2str(randind(2)) '.JPG'');'])
cue2=Screen('MakeTexture',wd,tmp);

%.................... The feedback 
eval(['tmp=imread(''imgs' filesep '10ct_x.bmp'');'])
frowny=Screen('MakeTexture',wd,tmp);
eval(['tmp=imread(''imgs' filesep '10ct.bmp'');'])
smiley=Screen('MakeTexture',wd,tmp);
text_fb_too_slow = 'Zu langsam!';

% position to display feedback too slow
xpos_fb = x0+xl0* .2; ypos_fb = y0+yl0* .3; 

% position to display exploratory button
xpos_expl = x0+xl0* .2; ypos_expl = y0+yl0* .1; 

    
% write outcome explicitly, too 
txt_win   = 'Gewonnen!';
cent_win  = ['+ 10 Cent'];
col_gr    = green;
txt_loss  = 'Verloren!';
cent_loss = ['- 10 Cent'];
col_re    = red;
txt_help  = 'Diese wäre richtig gewesen';
col_wh  = white;

% text for helpbutton
txt_expl = 'Hinweis [Leertaste]';

% arrows 
eval(['tmp=imread(''imgs' filesep 'arrows.tif'');'])
arrow=Screen('MakeTexture',wd,tmp);
arrowsquare(1,:)=[wdw*.02 wdh*.92 wdw*.16 wdh*.98];

% instructions positions
addpath('instr_funcs');
yposm = 'center'; 
yposb = .8*wdh; 
ypost = .1*wdh; 
ypostt=.05*wdh;

% monitor frame rate
[monitorFlipInterval nrValidSamples stddev] = Screen('GetFlipInterval', wd);

% Set up alpha-blending for smooth (anti-aliased) lines
Screen('BlendFunction', wd, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
 