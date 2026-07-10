clear; clc;
% =========================================================
% 纯文本模式：批量在CSV文件开头增加6行并覆盖原文件（递归所有子文件夹，格式不变）
% =========================================================

% 1. 设置文件夹路径 (请根据实际情况修改)
folderPath = "E:\牙菌斑项目\牙菌斑SERS光谱\阳性+\1-20"; 

% 2. 递归获取该文件夹及所有子文件夹下，所有以 'SP_' 开头的 csv 文件
% ** 通配符表示匹配任意层级的子目录，包含当前文件夹本身
filePattern = fullfile(folderPath, '**', 'SP_*.csv');
csvFiles = dir(filePattern);

if isempty(csvFiles)
    disp('未找到符合要求的CSV文件，请检查文件夹路径。');
    return;
end

fprintf('共找到 %d 个文件，准备开始处理...\n', length(csvFiles));

% ========== 可配置：定义要插入的前6行内容 ==========
% 如需插入6行空行，替换为：insertLines = strings(6, 1);
insertLines = [
    "新增行1"
    "新增行2"
    "新增行3"
    "新增行4"
    "新增行5"
    "新增行6"
];
% ==================================================

% 3. 遍历每个文件进行处理
for i = 1:length(csvFiles)
    currentFile = fullfile(csvFiles(i).folder, csvFiles(i).name);
    
    try
        % 纯文本逐行读取，完全保留原文件的逗号、Tab等格式
        textLines = readlines(currentFile);
        
        % 在文件最开头插入6行内容
        modifiedLines = [insertLines; textLines];
        
        % 原样覆盖写入原文件
        writelines(modifiedLines, currentFile);
        
        fprintf('成功处理并覆盖: %s\n', currentFile);
        
    catch ME
        fprintf('❌ 处理 %s 时出错: %s\n', currentFile, ME.message);
    end
end

disp('========================================');
disp('所有文件已成功在开头增加6行，格式原封不动！');