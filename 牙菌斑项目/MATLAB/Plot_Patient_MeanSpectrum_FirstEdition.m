%% Plot_Patient_MeanSpectrum_FirstEdition.m
% First Edition - Patient-level mean spectrum visualization
clear; clc; close all;

rootFolder = 'E:\牙菌斑项目\牙菌斑SERS光谱\阳性+';
lambda = 1e3;

csvFiles = dir(fullfile(rootFolder,'**','SP_*.csv'));
assert(~isempty(csvFiles),'No SP_*.csv files found.');

patientMap = containers.Map;
for i=1:numel(csvFiles)
    remain = erase(csvFiles(i).folder,rootFolder);
    parts = strsplit(remain,filesep);
    parts = parts(~cellfun('isempty',parts));
    pid = parts{1};
    if ~isKey(patientMap,pid), patientMap(pid)={}; end
    t = patientMap(pid);
    t{end+1}=fullfile(csvFiles(i).folder,csvFiles(i).name);
    patientMap(pid)=t;
end

patients = sort(keys(patientMap));
perPage = 5;

for pg = 1:ceil(numel(patients)/perPage)
    figure('Color','w','Position',[100 50 900 1300]);
    tiledlayout(perPage,1,'TileSpacing','compact','Padding','compact');

    for ii=1:perPage
        idx=(pg-1)*perPage+ii;
        if idx>numel(patients), break; end

        pid=patients{idx};
        files=patientMap(pid);

        Raw=[]; shift=[];
        for k=1:numel(files)
            Raw(:,k)=readmatrix(files{k},'Range','H294:H1025');
            if isempty(shift)
                shift=readmatrix(files{k},'Range','D294:D1025');
            end
        end
        Raw=Raw';

        Smooth=sgolayfilt(Raw,3,7,[],2);
        [Corrected,~]=airPLS(Smooth,lambda,2,0.05);
        SNV=snv(Corrected')';
        Mean=mean(SNV,1);
        SD=std(SNV,0,1);
        SEM=SD/sqrt(size(SNV,1));

        sf=max(Mean);
        Mean=Mean/sf; SEM=SEM/sf;

        % SNR
        snrlist=zeros(size(SNV,1),1);
        for s=1:size(SNV,1)
            sig=max(SNV(s,:))-min(SNV(s,:));
            noise=std(diff(SNV(s,:)));
            snrlist(s)=sig/(noise+eps);
        end
        meanSNR=mean(snrlist);

        % correlation
        corrv=zeros(size(SNV,1),1);
        for s=1:size(SNV,1)
            c=corrcoef(SNV(s,:),Mean*sf);
            corrv(s)=c(1,2);
        end
        meanCorr=mean(corrv);

        nexttile; hold on;
        x=shift';
        fill([x fliplr(x)],[Mean+SEM fliplr(Mean-SEM)],...
            [0.4 0.7 1],'FaceAlpha',0.25,'EdgeColor','none');
        plot(x,Mean,'b','LineWidth',1.8);
        box on; grid on;
        xlim([min(x) max(x)]);
        xlabel('Raman Shift (cm^{-1})');
        ylabel('Norm. Int.');
        title(['Patient ' pid],'FontWeight','bold');

        txt=sprintf(['Patient: %s\n',...
                     'n = %d spectra\n',...
                     'Mean SNR = %.2f\n',...
                     'Mean Corr = %.3f'],...
                     pid,size(SNV,1),meanSNR,meanCorr);
        text(0.98,0.95,txt,'Units','normalized',...
            'HorizontalAlignment','right',...
            'VerticalAlignment','top',...
            'BackgroundColor','w','Margin',4,...
            'FontSize',9);
    end

    exportgraphics(gcf,fullfile(rootFolder,...
        sprintf('PatientMean_Page_%02d.png',pg)),...
        'Resolution',600);
end

disp('Done.');
