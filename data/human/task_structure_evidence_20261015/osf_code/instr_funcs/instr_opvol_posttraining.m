i=0; clear tx ypos func;
func{1}=[];

keysOfInterest = zeros(1,256);
keysOfInterest(KbName(instrbackward)) = 1;
keysOfInterest(KbName(instrforward)) = 1;
keysOfInterest(KbName(stopsession)) = 1;
KbQueueCreate(deviceIndex, keysOfInterest);

i=i+1; 
	ypos{i}=yposm;
	tx{i}='Die Übung ist jetzt zu Ende.';
    
i=i+1; 
	ypos{i}=yposm;
	tx{i}='Wie bei den meisten Gewinnspielen mischt der Zufall auch mit und es kann passieren, dass eine Karte die zunächst häufig gewonnen hat schlechter wird. Im Gegenzug wird dann die andere Karte besser. \n\n Achte stets wachsam auf solche Veränderungen, um so viel Geld wie möglich zu gewinnen!';

if regularinstr==1
    
i=i+1; 
	ypos{i}=yposm;
	tx{i}='Diese Wechsel sind immer regelmäßig.';  
   i=i+1;
	ypos{i}=yposm;
	tx{i}='Achte darauf, ob du Regelmäßigkeiten erkennen kannst, darin, wann eine Karte schlechter, und die andere Karte besser wird.  \n\n Achte stets wachsam auf solche Veränderungen, um so viel Geld wie möglich zu gewinnen!';

elseif regularinstr==2
i=i+1; 
	ypos{i}=yposm;
	tx{i}='Diese Wechsel sind immer unregelmäßig.  \n\n Achte stets wachsam auf Veränderungen, wann eine Karte besser oder schlechter ist, um so viel Geld wie möglich zu gewinnen!';
end

i=i+1; 
	ypos{i}=yposm;
	tx{i}='Denke daran, dass Dir die Hinweiskarte helfen kann, herauszufinden, welche Karte in einem Durchgang gewonnen hätte, ohne dass Du riskieren musst, in diesem Durchgang 10ct zu verlieren.';
    

i=i+1; 
	ypos{i}=yposm;
	tx{i}='Gleich hast Du für dieses Gewinnspiel etwa 30 Minuten Zeit. Deinen Gewinn bekommst Du danach ausgezahlt!';
    
i=i+1; 
	ypos{i}=yposm;
	tx{i}='Wir möchten jetzt sicher stellen, dass Du alles verstanden hast. Bitte erkläre, was Deine Aufgabe in diesem Gewinnspiel ist.';
    
if doeeg
i=i+1; 
	ypos{i}=yposm;
	tx{i}='Außerdem möchte wir Sie bitten während die Karten oder der Gewinn gezeigt werden nicht zu blinzeln. Bitte blinzeln Sie nur nach dem der Gewinn nicht mehr sichtbar ist!';
end

instr_display;