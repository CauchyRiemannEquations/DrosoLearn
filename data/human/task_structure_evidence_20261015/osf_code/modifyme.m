%----------------------------------------------------------------------------
%        MAIN FILE TO EDIT
%
%        This is the only file that should demand any changes!!!
%
%----------------------------------------------------------------------------
fprintf('............ Setting basic parameters according to \n')
fprintf('............            MODIFYME.M\n'); fprintf('............ \n')

debug       = 0; % do debug mode, small screen (see setup.m)

doinstr     = 0; % 0: no instructions, just the experiment inside or outside the scanner
                 % 1: instructions and training, can be performed inside or
                 % outside the scanner 
%----------------------------------------------------------------------------
%        If set doscanner=1 exp will wait for the first trigger 
%        of the MR scanner and keys are different
%
%        If set doeeg=1 exp will wait for a button press (experimenter) to start 
%        and send markers, keys are different
%----------------------------------------------------------------------------
doscanner = 0; % 0: outside scanner the scanner,  1: inside the scanner 

doeeg     = 0; % do experiment and record eeg = 1 
doeda     = 0; % do experiment and record eda = 1 
%----------------------------------------------------------------------------
%        To save or not to save
%        This should ALWAYS be set to 1 when doing experiments obviously
%----------------------------------------------------------------------------
dosave = 1;      % save output? 

%----------------------------------------------------------------------------
%        Patient Information 
%--------------------------------------------------------------------------
type = '10';      % age tag 10=young adult

subjn = '99';    % Subject Number. Aufsteigend, fortlaufend 

payment = 1;      % is this subject being paid / should payment info be displayed


%-------------------------------------- --------------------------------------
%        EXPERIMENT VERSION 
%			PLEASE check this is correct! 
%----------------------------------------------------------------------------
expversion = 'TEMP_OPVOL_3.0_190611_TUD';
