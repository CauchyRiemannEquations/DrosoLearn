fprintf('............. Displaying instructions \n');
i=0; clear tx ypos func;
func{1}=[];

keysOfInterest = zeros(1,256);
keysOfInterest(KbName(instrbackward)) = 1;
keysOfInterest(KbName(instrforward)) = 1;
keysOfInterest(KbName(stopsession)) = 1;
KbQueueCreate(deviceIndex, keysOfInterest);

if doscanner | doeeg
    i=i+1; 
	ypos{i}=yposm;
	tx{i}=['Willkommen zu diesem Gewinnspiel.\n\n Wir werden es dir jetzt erklären.\n\n Benutze bitte die rechte Taste, um vorwärts zu blättern und die linke Taste, um zurück zu blättern.'];
else
    i=i+1; 
	ypos{i}=yposm;
	tx{i}=['Willkommen zu diesem Gewinnspiel.\n\n Wir werden es Dir jetzt erklären.\n\n Benutze bitte die rechte Pfeiltaste, um vorwärts zu blättern und die linke Pfeiltaste, um zurück zu blättern.'];
end
i=i+1; 
	ypos{i}=yposm;
	tx{i}=['Dieses Gewinnspiel ist ein einfaches Kartenspiel. Es geht darum, so viel Geld wie möglich zu gewinnen.\n\n Deinen Geldgewinn bekommst Du am Ende ausgezahlt!'];

i=i+1; 
	ypos{i}=yposm;
	%tx{i}=['In jedem Durchgang wählst Du zwischen zwei Spielkarten und der Hinweiskarte.'];
    tx{i}=['In jedem Durchgang wählst Du zwischen drei Karten, nämlich zwischen zwei Spielkarten oder einer Hinweiskarte'];
if  doscanner | doeeg
    i=i+1; 
	ypos{i}=ypost;
	tx{i}=['Immer, wenn Du die Karten siehst, hast Du 3 Sekunden Zeit, eine auszuwählen.\n\n Dazu drückst Du bitte die linke Taste für die linke Karte und die rechte Taste für die rechte Karte.'];
    func{i}='Screen(''DrawTexture'',wd,cue4, [],box(1,:)); Screen(''DrawTexture'',wd,cue3, [],box(2,:));getleftrightarrow;';
else
    i=i+1; 
	ypos{i}=ypost;
	tx{i}=['Immer, wenn Du diesen Bildschirm siehst, hast Du 3 Sekunden Zeit, eine Auswahl zu treffen.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n Dazu drückst Du bitte die Taste f für die linke Karte, die Taste j für die rechte Karte, und die Lehrtaste für die Hinweiskarte.'];
    func{i}='draw_hints; Screen(''DrawTexture'',wd,cue4, [],box(1,:)); Screen(''DrawTexture'',wd,cue3, [],box(2,:));getleftrightarrow;';
end
    
i=i+1; 
	ypos{i}=ypost;
	tx{i}=['Falls Du in dem Durchgang eine Spielkarte gewählt hast, bekommst Du eine Rückmeldung über Geldgewinn oder Verlust. \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n Entweder gewinnst Du mit der gewählten Karte 10 Cent...'];
    func{i}='draw_hints; Screen(''DrawTexture'',wd,squareframe,[],boxl(1,:)); Screen(''DrawTexture'',wd,cue4, [],box(1,:)); Screen(''DrawTexture'',wd,cue3, [],box(2,:));Screen(''DrawTexture'',wd,smiley,[],box_center);getleftrightarrow;';
    
i=i+1; 
	ypos{i}=ypost;
	tx{i}=[' \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n ... oder Du verlierst mit der ausgewählten Karte 10 Cent.'];
    func{i}='draw_hints; Screen(''DrawTexture'',wd,squareframe,[],boxl(1,:)); Screen(''DrawTexture'',wd,cue4, [],box(1,:)); Screen(''DrawTexture'',wd,cue3, [],box(2,:));Screen(''DrawTexture'',wd,frowny,[],box_center);getleftrightarrow;';
    
i=i+1; 
	ypos{i}=yposm;
	tx{i}=['Wie es bei einem Gewinnspiel so üblich ist, ist auch ein bisschen Glück dabei, denn keine Spielkarte gewinnt oder verliert immer.'];

i=i+1; 
	ypos{i}=yposm;
	tx{i}=['Du sollst die Karte auswählen, welche am häufigsten gewinnt, damit Du insgesamt möglichst viel Geld gewinnst, denn Deinen Gewinn bekommst Du am Ende ausgezahlt.'];
    
    %% added: possibility of pressing space to see which card wins this round
i=i+1; 
	ypos{i}=yposm;
	tx{i}=['Du kannst aber in jedem Durchgang auch die Hinweiskarte wählen, indem Du die Leertaste drückst. Sie hilft dir, denn Du kannst mit ihr den Hinweis bekommen, welche Karte im aktuellen Versuch die richtige gewesen wäre. \n\n In diesem Fall gewinnst Du keine 10 Cent, kannst aber auch keine 10 Cent verlieren.'];
    
    
i = i+1;
    ypos{i}=ypost;
    tx{i}=['Nach Drücken der Leertaste siehst Du folgendes Bild: \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n Die markierte Spielkarte wäre im aktuellen Versuch die richtige Wahl gewesen.'];
    func{i}='Screen(''FillRect'', wd, bluerect,CenterRectOnPoint([0 0 420 120], wdw/2, ypos_expl - 50)); draw_hints; Screen(''DrawTexture'',wd,squareframe,[],boxl(1,:));Screen(''DrawTexture'',wd,cue4, [],box(1,:)); Screen(''DrawTexture'',wd,cue3, [],box(2,:)); DrawFormattedText(wd,txt_help,''center'', box_center(2),col_wh); getleftrightarrow;';   
    %%
if doeeg
    i=i+1; 
	ypos{i}=yposm;
	tx{i}='Außerdem möchte wir Dich bitten während die Karten oder der Gewinn gezeigt werden nicht zu blinzeln. Bitte blinzeln Sie nur nach dem der Gewinn nicht mehr sichtbar ist!';
end

i=i+1; 
	ypos{i}=yposm;
	tx{i}=['Die Hinweiskarte hilft Dir also, Informationen zu gewinnen, welche Karte gerade gewinnt oder verliert, ohne dass Du riskierst in diesem Durchgang 10 cent zu verlieren.'];
    
i=i+1; 
	ypos{i}=yposm;
	tx{i}='Lass uns das kurz üben. \n\n Bitte versuche in der Übung, alle drei Tasten, also Spielkarten und Hinweiskarte zu benutzen um ihre Funktion zu verstehen. Beachten Sie den Unterschied zwischen guten und schlechten Auswahlmöglichkeiten.';
    
instr_display;
