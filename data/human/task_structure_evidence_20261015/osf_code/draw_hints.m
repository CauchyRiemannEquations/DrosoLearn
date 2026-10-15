% Draw recangle behind the text
centBar = CenterRectOnPoint([0 0 400 100], wdw/2, ypos_expl - 50);
Screen('FillRect', wd, yellowrect, centBar);

% Show hint instruction (space button for correct choice)
DrawFormattedText(wd,txt_expl,'center', ypos_expl-40, bluetxt);