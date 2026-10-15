i=0; clear tx ypos func;
func{1}=[];
fprintf('............. payout\n');
payout = round(10*Z.cent_per_point*(sum(Rew==1) - sum(Rew==-1)))/10;
if payout <= minpayout;
   payout = minpayout;
end

if payout >= maxpayout;
    payout = maxpayout;
end
if payment
	i=i+1;
	ypos{i}=yposm;
	tx{i}= ['du hast insgesamt ' num2str(payout) ' Euro verdient!\n\n Vielen Dank für deine Teilnahme!'];
end
display_payment; 

