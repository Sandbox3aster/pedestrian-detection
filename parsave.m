function parsave(fname,varargin)
	numvars=numel(varargin);
	for i=1:numvars
	   eval([inputname(i+1),'=varargin{i};']);  
	end
	save([fname '.mat'],inputname(2),'-mat');
	for i = 2:numvars    
		save([fname '.mat'],inputname(i+1),'-mat','-append');
	end
end