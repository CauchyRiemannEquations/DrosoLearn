%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% preps %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

aborted=0; % if this parameter is set to one, things will abort. 

% make sure we use different random numbers
rand('twister',sum(1000*clock));

%....................... Saving 
namestring_long  = ['OPVOL_' type '_'  subjn datestr(now,'_yymmdd_HHMM')]; % detailed name string
namestring_short = ['OPVOL_' type '_' subjn];                              % simplified name string 

if doinstr; namestring_short = [namestring_short '_training'];namestring_long = [namestring_long '_training'];end

if exist('data')~=7; eval(['!mkdir data']); end % make 'data' folder if dosn't exist

if dosave 
	fprintf('............ Data will be saved as                              \n');
	fprintf('............ %s and %s \n',namestring_long,namestring_short);
	fprintf('............ in the folder ''data''\n');
end

% setup variables

A=NaN(1,Z.Ntrials); 
C=NaN(1,Z.Ntrials);
S=NaN(1,Z.Ntrials);
R=NaN(1,Z.Ntrials);

% define random numbers
% left or right -- change ones to rand for random location assignment
random_lr               = (ones(1,Z.Ntrials)>0.5)+1;	

%define blockorder according to subject number (even vs. odd)
blockorder=mod(str2num(subjn)+1,2);

%%define instruction condition according to subjectnumber 
regularinstr = 1; % dumvect(str2num(subjn(2))); %change this to 2 for irregular instractions


