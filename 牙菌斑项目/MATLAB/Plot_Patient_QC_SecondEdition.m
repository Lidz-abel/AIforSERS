%% Plot_Patient_QC_SecondEdition.m
% Second Edition
% Adds:
% 1. QC filtering by correlation
% 2. QC dashboard (SNR / Correlation)
% 3. Peak labels
% Requires: airPLS.m, snv.m

clear;clc;close all;

rootFolder='E:\牙菌斑项目\牙菌斑SERS光谱\阳性+';
lambda=1e3;
corrThreshold=0.85;

csvFiles=dir(fullfile(rootFolder,'**','SP_*.csv'));
patientMap=containers.Map;

for i=1:numel(csvFiles)
    remain=erase(csvFiles(i).folder,rootFolder);
    p=strsplit(remain,filesep);
    p=p(~cellfun('isempty',p));
    id=p{1};
    if ~isKey(patientMap,id), patientMap(id)={}; end
    t=patientMap(id);
    t{end+1}=fullfile(csvFiles(i).folder,csvFiles(i).name);
    patientMap(id)=t;
end

patients=sort(keys(patientMap));

for k=1:numel(patients)

    files=patientMap(patients{k});
    Raw=[]; shift=[];

    for i=1:numel(files)
        Raw(:,i)=readmatrix(files{i},'Range','H294:H1025');
        if isempty(shift)
            shift=readmatrix(files{i},'Range','D294:D1025');
        end
    end

    Raw=Raw';
    Smooth=sgolayfilt(Raw,3,7,[],2);
    [Corrected,~]=airPLS(Smooth,lambda,2,0.05);
    X=snv(Corrected')';

    mean0=mean(X,1);

    C=zeros(size(X,1),1);
    for i=1:size(X,1)
        cc=corrcoef(X(i,:),mean0);
        C(i)=cc(1,2);
    end

    keep=C>=corrThreshold;
    X2=X(keep,:);

    Mean=mean(X2,1);
    SD=std(X2,0,1);
    SEM=SD/sqrt(size(X2,1));

    sf=max(Mean);
    Mean=Mean/sf;
    SEM=SEM/sf;

    snr=zeros(size(X2,1),1);
    for i=1:size(X2,1)
        snr(i)=(max(X2(i,:))-min(X2(i,:)))/(std(diff(X2(i,:)))+eps);
    end

    figure('Color','w','Position',[120 80 1200 800]);

    tiledlayout(2,2);

    %% Mean spectrum
    nexttile([1 2]); hold on;
    x=shift';
    fill([x fliplr(x)],[Mean+SEM fliplr(Mean-SEM)],...
        [0.5 0.75 1],'EdgeColor','none','FaceAlpha',0.3);
    plot(x,Mean,'b','LineWidth',2);

    peaks=[730 785 1002 1095 1332 1450 1660];
    for p=1:length(peaks)
        [~,idx]=min(abs(x-peaks(p)));
        text(x(idx),Mean(idx)+0.04,num2str(peaks(p)),...
            'Rotation',90,'FontSize',8,...
            'HorizontalAlignment','center');
    end

    title(['Patient ' patients{k}]);
    xlabel('Raman Shift (cm^{-1})');
    ylabel('Normalized');
    grid on; box on;

    txt=sprintf(['Total spectra : %d\n',...
                 'Remaining : %d\n',...
                 'Removed : %d\n',...
                 'Mean Corr : %.3f\n',...
                 'Mean SNR : %.2f'],...
                 size(X,1),size(X2,1),...
                 size(X,1)-size(X2,1),...
                 mean(C(keep)),mean(snr));

    text(0.985,0.97,txt,'Units','normalized',...
        'HorizontalAlignment','right',...
        'VerticalAlignment','top',...
        'BackgroundColor','w');

    %% SNR
    nexttile;
    histogram(snr,15);
    xlabel('SNR');
    ylabel('Counts');
    title('SNR Distribution');

    %% Correlation
    nexttile;
    histogram(C,15);
    hold on;
    xline(corrThreshold,'r--','Threshold');
    xlabel('Correlation');
    ylabel('Counts');
    title('Correlation QC');

    exportgraphics(gcf,fullfile(rootFolder,...
        ['QC_' patients{k} '.png']),'Resolution',600);

    close;
end

disp('QC finished.');
