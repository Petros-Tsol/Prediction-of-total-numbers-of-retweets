function [] = retweet_prediction()
	%training time in seconds
	indicator_time = 4*3600;
	%test time in seconds
	reference_time = 48*3600;
	
	%open a test file
	%test file must be in the specific format
	%each line (exept the very first) represents the tweet/retweet reference time in seconds and the average number of followers of the user who tweeted/retweeted
	%first line: <number of retweets> <average number of followers>
	%second line: 0 <average number of followers of source>
	%third line: <seconds passed between the time of tweet and this first retweet> <average number of followers of this user>
	%forth line: <seconds passed between the time of tweet and this second retweet> <average number of followers of this user>
	%fifth line: etc...
    T=dlmread('./test_data/test_message.txt');
	retweet_time = T(2:end,1);
	retweet_followers = T(2:end,2);
	
	ln_errors = linear_regression(indicator_time,reference_time,retweet_time);
	
    pp_errors = reinforced_poisson_process(indicator_time,reference_time,retweet_time);
	
	hp_notrain_errors = hawkes_process(indicator_time,reference_time,retweet_time,retweet_followers);

    params = hawkes_process_training(indicator_time,reference_time);	
    hp_train_errors = hawkes_process(indicator_time,reference_time,retweet_time,retweet_followers,params);
	
	
	disp(['Mean absolute percentage error for linear regression: ',num2str(100*mean(ln_errors)),'%']);
	disp(['Mean absolute percentage error for reinforced poisson process: ',num2str(100*mean(pp_errors)),'%']);
	disp(['Mean absolute percentage error for hawkes process without training: ',num2str(100*mean(hp_notrain_errors)),'%']);
	disp(['Mean absolute percentage error for hawkes process with training: ',num2str(100*mean(hp_train_errors)),'%']);

	
	plot(1+indicator_time/3600:reference_time/3600,100.*ln_errors);
	hold on;
	plot(1+indicator_time/3600:reference_time/3600,100.*pp_errors);
	hold on;
	plot(1+indicator_time/3600:reference_time/3600,100.*hp_notrain_errors);
	hold on;
	plot(1+indicator_time/3600:reference_time/3600,100.*hp_train_errors);
	hold on;
    title('Evolution of mean absolute percentage error up to reference time');
	xlabel('reference time');
	ylabel('Mean absolute percentage error %');
	legend('linear regression','reinforced poisson process','hawkes process without training','hawkes process training','Location','northwest');
	hold off;
end

function [errors] = linear_regression(indicator_time,reference_time,retweet_time)
	number_of_files = size(dir('./train_data/*.txt'),1);
	file_number=1;
	n=1;
	%the rows of this matrix represent a different training tweet
	%the columns represents the cumulative number of retweets of this tweet starting from indicator time and ending in reference time
	%the time step is one hour so the columns are indicator_time|indicator_time+1h|indicator_time+2h|....|reference_time
	retweets_matrix = zeros(number_of_files,(reference_time-indicator_time)/3600+1);
	while file_number <= 100
		%open the file
		filename = sprintf('./train_data/RT%d.txt',file_number);
		file_number = file_number + 1;

		T=dlmread(filename);
		
		for t=indicator_time:3600:reference_time
			retweets = size(find(T(3:end,1) <= t),1);
			retweets_matrix(n,(t-indicator_time)/3600+1) = retweets;
		end
		
		n=n+1;
	end
	
	%1st column of this matrix represents the b0 parameter
	%2nd column of this matrix represents the variance s^2
	%rows of this matrix represents the estimated parameters at a specific prediction time
	%so first row is indicator_time+1h, second row is indicator_time+2h etc until reference_time
	parameters = zeros((reference_time-indicator_time)/3600,2);
	
	predictions = zeros((reference_time-indicator_time)/3600,1);
	real_retweets = zeros((reference_time-indicator_time)/3600,1);
	errors = zeros((reference_time-indicator_time)/3600,1);
	for hour = 1+indicator_time/3600:reference_time/3600
		%equation 3 on paper page 6
		lse = @(b) sum((-log(retweets_matrix(:,hour-indicator_time/3600+1))+b+log(retweets_matrix(:,1))).^2);
		[beta0,~]=fminunc(lse,1);
		parameters(hour-indicator_time/3600,1) = beta0;
		
		sigma0 = var(-log(retweets_matrix(:,hour-indicator_time/3600+1))+beta0+log(retweets_matrix(:,1)),1);
		parameters(hour-indicator_time/3600,2) = sigma0;
		
		%prediction
		retweets_on_indicator_time = size(find(retweet_time(2:end)<=indicator_time),1);
		predictions(hour-indicator_time/3600,1) = round(exp(log(retweets_on_indicator_time) + beta0 + sigma0/2));
		real_retweets(hour-indicator_time/3600,1) = size(find(retweet_time(2:end)<=hour*3600),1);
		errors(hour-indicator_time/3600,1) = abs(real_retweets(hour-indicator_time/3600,1) - predictions(hour-indicator_time/3600,1))/abs(real_retweets(hour-indicator_time/3600,1));
	end
% 	figure();
% 	plot(1+indicator_time/3600:reference_time/3600,predictions(:));
% 	hold on;
% 	plot(1+indicator_time/3600:reference_time/3600,real_retweets(:));
% 	hold off;
% 	
% 	figure();
% 	bar(1+indicator_time/3600:reference_time/3600,100.*errors(:),0.1);
% 	disp(['Mean absolute percentage error: ',num2str(100*mean(errors(:))), ' %']);
end

function [errors] = reinforced_poisson_process(indicator_time,reference_time,retweet_time)
	%number of training messages
	n=0;
	for i=1:size(retweet_time)
		if retweet_time(i,1) <= indicator_time
			n=n+1;
		else
			break;
		end
	end

	%train_data(1): original tweet
	%train_data(2:n): retweets
	%train_data(n+1): indicator time
	%+1 it's the t* time, page 5 in paper
	train_data = retweet_time(1:n,1)+1;
	train_data = [train_data;indicator_time];


	k = 1:n;
	epsilon = 10;
	X = @(v) sum((train_data(2:n+1).^(1-v(2)) - train_data(1:n).^(1-v(2))).*(1+epsilon-epsilon*exp(-v(1))-exp(-v(1).*k(:)))) / ((1-v(2))*(1-exp(-v(1))));
	
	%c parameter, page 6 in paper, equation 18
	c = @(v) (n-1)/X(v);


	%calculate the sum in log likelihood, page 5 in paper, equation 16
	m = 1:n-1;
	Z = @(v) sum(log(1+epsilon-epsilon*exp(-v(1))-exp(-v(1).*m(:)))-v(2).*log(train_data(2:n)));

	%log-likelihood function, page 5 in paper, equation 16
	log_ll = @(v)(n-1)*log(c(v)/(1-exp(-v(1)))) + Z(v) - c(v)*X(v);
    
  
	%multiply log likelihood function with -1 to find the maximum
	log_ll_max = @(v) -1*log_ll(v);
	sol=fminsearch(log_ll_max, [0.05,2.5]);
	a_val = sol(1);
	g_val = sol(2);
	c = c([a_val,g_val]);
	
	%equation 27 and 26 page 6 on paper
	Y = @(t) ((1+epsilon-epsilon*exp(-a_val))*a_val*c*(indicator_time^(1-g_val)-t^(1-g_val)))/((1-g_val)*(1-exp(-a_val))) - n*a_val - log(1+epsilon-epsilon*exp(-a_val)-exp(-a_val*n));
	N_t = @(t)(log(1+exp(Y(t))) - Y(t) - log(1+epsilon-epsilon*exp(-a_val)) - a_val)/a_val;
	
	%predict the number of retweets for the next hours
	%first column are the predicted retweets
	%second column are the actual retweets
	%third column are the absolute error
	retweets=zeros((reference_time-indicator_time)/3600,2);
	for i=1+indicator_time/3600:reference_time/3600
		actual_retweets = size(find(retweet_time <= i*3600),1);
		
		retweets(i-indicator_time/3600,1)=round(real(N_t(i*3600)));
		retweets(i-indicator_time/3600,2)=actual_retweets;
		retweets(i-indicator_time/3600,3) = abs(retweets(i-indicator_time/3600,1)-retweets(i-indicator_time/3600,2))/abs(retweets(i-indicator_time/3600,2));
	end
	% figure();
	% plot(1+indicator_time/3600:reference_time/3600,retweets(:,1));
	% hold on;
	% plot(1+indicator_time/3600:reference_time/3600,retweets(:,2));
	% hold off;
	
	% figure();
	% bar(1+indicator_time/3600:reference_time/3600,100.*retweets(:,3),0.1);
	% disp(['Mean absolute percentage error: ',num2str(100*mean(retweets(:,3))),'%']);
	errors = retweets(:,3);
end


function [errors] = hawkes_process(indicator_time,reference_time,retweet_time, retweet_followers,parameters)
	%period of time window in seconds
	time_window = 1*3600;
	%window slide in seconds
	window_slide = (1/4)*3600;

	%first position, last position
	fpos = 2; lpos=1;
	%start time, end time
	stime = 0; etime = time_window;
	
	n=1;
	%equation 4 on paper, page 4
	while stime + time_window <= indicator_time
		lpos = find(retweet_time>=etime,1);
		
		if isempty(lpos) == 1
			break;
		else
			iir(n) = instantaneous_infectious_rate(retweet_time(1:lpos-1),retweet_followers(1:lpos-1),stime,etime,lpos-fpos);	
			iir_time(n) = (stime+etime)/2;
			
			n = n+1;
			stime = stime + window_slide;
			etime = stime + time_window;
			fpos = find(retweet_time>=stime,1);
		end
	end	

    %minimize the infectious rate, equation 6 on paper, page 4
	if (nargin == 4)
		infectious_rate = @(c,tdata) c(1) * (1-c(2)*sin((2*pi)*(tdata+c(3)))).*exp(-tdata/c(4));
		options = optimoptions('lsqcurvefit','Algorithm','trust-region-reflective');

		c = lsqcurvefit(infectious_rate,[0,0.2,0,2],iir_time./86400,iir,[-inf,-1,-inf,0.5],[+inf,1,+inf,20],options);	
	elseif (nargin == 5)
		r0 = parameters(1);
		f0 = parameters(2);
		tm = parameters(3);
		infectious_rate = @(c,tdata) c * (1-r0*sin((2*pi)*(tdata+f0))).*exp(-tdata/tm);

        options = optimoptions('lsqcurvefit','Algorithm','trust-region-reflective');
        c = lsqcurvefit(infectious_rate,0,iir_time./86400,iir,[],[],options);
	end

	%Volterra equation solution based on book Numerical Recipes	
	%step of integration in seconds and number of steps
	h = 6*60;
	N = round((reference_time - indicator_time)/h);
	
	%average followers of retweeters
	lpos = find(retweet_time>=indicator_time,1);
	avg_foll = mean(retweet_followers(2:lpos-1));
	
	%retweet rate lambda(t) on paper
	retweet_rate = zeros(N,1);
	%calculate the contribution of observed tweets at the end of observation time, that's g0 in Numerical Recipes
	retweet_rate(1,1) = contribution_observed_tweets(infectious_rate(c,indicator_time/86400),retweet_time(1:lpos-1),retweet_followers(1:lpos-1),indicator_time);
	%disp(retweet_rate(1,1));
	time = indicator_time + h;
	
	
	%first column are the predicted retweets
	%seconds column are the actual retweets
	%third column are the absolute difference
	retweets=zeros((reference_time-indicator_time)/3600,3);
	retweets_till_now = size(find(retweet_time <= indicator_time),1) - 1;
	cou=1;
	
	for i=1:N
		right_part = 0;
		for j=1:i-1
			right_part = right_part + retweet_rate(j,1)*reaction_time_distribution(time - (indicator_time + j*h));
		end
		
		%right part and left part of equation whichs solves a Volterra integral equation from Numerical Recipes
		right_part = right_part + 0.5 * reaction_time_distribution(time) * retweet_rate(1,1);
		right_part = right_part*h*avg_foll*infectious_rate(c,time/86400) + contribution_observed_tweets(infectious_rate(c,time/86400),retweet_time(1:lpos-1),retweet_followers(1:lpos-1),time);
		
		left_part = 1 - 0.5*h*avg_foll*infectious_rate(c,time/86400)*reaction_time_distribution(0);
		
		retweet_rate(i+1,1) = right_part/left_part;
		if mod(time,3600) == 0
			retweets(cou,1) = retweets_till_now + floor(sum(retweet_rate)*h);
			retweets(cou,2) = size(find(retweet_time <= time),1) - 1;
			retweets(cou,3) = abs(retweets(cou,1)-retweets(cou,2))/abs(retweets(cou,2));
			cou=cou+1;
		end
		time = time + h;
	end
% 	figure();
% 	plot(1+(indicator_time/3600):reference_time/3600,retweets(:,1));
% 	hold on;
% 	plot(1+(indicator_time/3600):reference_time/3600,retweets(:,2));
% 	hold off;
% 	
% 	figure();
% 	bar(1+(indicator_time/3600):reference_time/3600,100.*retweets(:,3),0.1);
% 	disp(['Mean absolute percentage error: ',num2str(100*mean(retweets(:,3))),'%']);

	errors = retweets(:,3);
end

function [react_time] = reaction_time_distribution(s)
	%reaction time distribution, page 4 in paper, equation 3
	theta = 0.242;
	c0 = 6.49 * 10^(-4);
	s0 = 300;
	
	if s<0
		react_time = 0;
	elseif s<=s0
		react_time = c0;
	else
		react_time = c0*(s/s0)^(-1-theta);
	end
end

function [react_time_integral] = reaction_time_distribution_integral(t)
	theta = 0.242;
	c0 = 6.49 * 10^(-4);
	s0 = 300;

	if t < 0
		react_time_integral = 0;
	elseif t <= s0
		react_time_integral = c0*t;
	else
		react_time_integral = c0*s0 + c0*((s0-s0*(t/s0)^(-theta)))/theta;
	end
end

function [p_t] = instantaneous_infectious_rate(tweets,followers,start_time,end_time,retweet_count)
	sum = 0;
	for i=1:size(tweets,1)
		r1 = reaction_time_distribution_integral(end_time-tweets(i));
		r2 = reaction_time_distribution_integral(start_time-tweets(i));
		sum = sum + followers(i)*(r1 - r2);
	end
	p_t = retweet_count/sum;
end

function [cont_obs_tw] = contribution_observed_tweets(irr,retweet_time,retweet_followers,time)
	%contribution of observed tweets, equation 9 on paper page 6, f(t) on pager, g(t) on Numerical Recipes
	sum = 0;	
	for i=1:size(retweet_followers,1)
		sum = sum + retweet_followers(i)*reaction_time_distribution(time - retweet_time(i));
	end
	cont_obs_tw = sum * irr;
end

function [params] = hawkes_process_training(indicator_time,reference_time)
	%number of files in directory
	number_of_files = 100; 
	
	%indexing matrix
	%first column index on events matrix
	%second column number of retweets
	%third column time of tweet
	%events_indeces = zeros(number_of_files,3);
    events_indeces = [];
	
	%events matrix
	%first column time of (re)tweet
	%second column followers of person who (re)tweeted
	events = [];
	start = 1;
    for i=1:number_of_files
		filename = sprintf('./train_data/RT%d.txt',i);
		T=dlmread(filename);

		events_indeces(i,1) = start;
		events_indeces(i,2:3) = T(1,:);
        
		events = [events;T(2:end,:)];
		start = start + size(T,1)-1;
    end
	
	params = fminsearch(@(x) minimize_function(x,events_indeces,events,indicator_time,reference_time),[0.2,0.1,2]);
    %disp(params);
end

function [error] = minimize_function(x,events_indeces,events,obs_time,pred_time)
	%period of time window in seconds
	time_window = 1*3600;
	%window slide in seconds
	window_slide = 0.25*3600;
	
	total_errors=zeros(size(events_indeces,1),1);
	%for every training tweet
	parfor tweet=1:size(events_indeces,1)
		retweet_time = events(events_indeces(tweet,1):events_indeces(tweet,1)+events_indeces(tweet,2),1);
		retweet_followers = events(events_indeces(tweet,1):events_indeces(tweet,1)+events_indeces(tweet,2),2);
		
		%first position, last position
		fpos = 2; lpos=1;
		%start time, end time
		stime = 0; etime = time_window;

		n=1;
		
		iir = 0;
		iir_time = 0;
		
		%if there is not a retweet in observation time, go to the next training tweet
        %OR if there is not a retweet in test time, go to the next training tweet
		if (retweet_time(2,1) > obs_time) || isempty(find(retweet_time > obs_time,1)) == 1
			continue;
		end
		
		%equation 4 on paper, page 4
		while stime + time_window <= obs_time
			lpos = find(retweet_time>=etime,1);
			
			if isempty(lpos) == 1
				break;
			else
				iir(n) = instantaneous_infectious_rate(retweet_time(1:lpos-1),retweet_followers(1:lpos-1),stime,etime,lpos-fpos);	
				iir_time(n) = (stime+etime)/2;

				n = n+1;
				stime = stime + window_slide;
				etime = stime + time_window;
				fpos = find(retweet_time>=stime,1);
			end
		end
		
		%equation 5 on paper, page 4
		t0 = events_indeces(tweet,3);
		infectious_rate = @(c,tdata) c(1) * (1-x(1)*sin((2*pi)*((tdata+t0)+x(2)))).*exp(-(tdata-t0)/x(3));
		
		%minimize the infectious rate, equation 6 on paper, page 4
		options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
		c = lsqcurvefit(infectious_rate,0,iir_time./86400,iir,[],[],options);
		
		%step of integration in seconds and number of steps
		h = 6*60;
		N = round((pred_time - obs_time)/h);

		%average followers of retweeters
		lpos = find(retweet_time>=obs_time,1);
		avg_foll = mean(retweet_followers(2:lpos-1));

		%retweet rate lambda(t) on paper
		retweet_rate = zeros(N,1);
		%calculate the contribution of observed tweets at the end of observation time, that's g0 in Numerical Recipes
		retweet_rate(1,1) = contribution_observed_tweets(infectious_rate(c,obs_time/86400),retweet_time(1:lpos-1),retweet_followers(1:lpos-1),obs_time);
		time = obs_time + h;
		
		for i=1:N
			right_part = 0;
			for j=1:i-1
				right_part = right_part + retweet_rate(j,1)*reaction_time_distribution(time - (obs_time + j*h));
			end

			%right part and left part of equation whichs solves a Volterra integral equation from Numerical Recipes
			right_part = right_part + 0.5 * reaction_time_distribution(time) * retweet_rate(1,1);

			right_part = right_part*h*avg_foll*infectious_rate(c,time/86400) + contribution_observed_tweets(infectious_rate(c,time/86400),retweet_time(1:lpos-1),retweet_followers(1:lpos-1),time);

			left_part = 1 - 0.5*h*avg_foll*infectious_rate(c,time/86400)*reaction_time_distribution(0);

			retweet_rate(i+1,1) = right_part/left_part;
			time = time + h;
		end
		pred_err = 0;
		n=1;
		
		%number of steps in a time window
		p = floor(time_window/h);
		for start_time=obs_time:time_window:pred_time
			end_time = start_time + time_window;			
			if end_time > pred_time
				break;
			end
			
			real_retweets = size(find(retweet_time>=start_time & retweet_time<=end_time),1);
			predicted_retweets = sum(retweet_rate(n:n+p,1))*h;
			pred_err = pred_err + abs(real_retweets-predicted_retweets);
			n=n+p;
		end
		total_errors(tweet,1) = pred_err;
	end
	error = median(total_errors);
end
