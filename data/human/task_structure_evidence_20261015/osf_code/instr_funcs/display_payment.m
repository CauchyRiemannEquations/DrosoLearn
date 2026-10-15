if length(func)<i;func{i}=[];end
Pages=i;
page=1;
Screen('TextSize',wd,txtsize);
DrawFormattedText(wd,tx{page},'center',ypos{page},txtcolor,40,[],[]); %1.3);
T.onset_payment = Screen('Flip',wd);
if page>Pages;end


