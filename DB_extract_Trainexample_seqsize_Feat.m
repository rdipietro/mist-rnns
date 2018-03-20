% NARX�� �̿��� regression �ϴ� �Լ�
% NARX�� �Է� �Ӹ� �ƴ϶� ����� feedback �Ͽ�, �̷��� data�� ���� �ϴ� �Լ���
clc; close all; clear ;
cd('E:\OneDrive_Hanyang\����\EMG_maker_regression\�ڵ�')
% ���� ����
N_mark = 28;
N_sub = 21;
Idx_sub = 1 : N_sub;
Idx_trl = 1 : 15;
Idx_sub4train = 1 : 19;
Idx_sub_4testing = 1; %  subejct dependent�� �Ұ��, train�� ������ �ε���
Idx_trl4train = 1 : 5;
Label_mark = {'central down lip';'central nose';'central upper lip';'head 1';'head 2';'head 3';'head 4';'jaw';'left central lip';'left cheek';'left dimple';'left down eye';'left down lip';'left eyebrow inside';'left eyebrow outside';'left nose';'left upper eye';'left upper lip';'right central lip';'right cheek';'right dimple';'right down eye';'right down lip';'right eyebrow inside';'right eyebrow outside';'right nose';'right upper eye';'right upper lip'};
delay = 1;
% neuronsHiddenLayer = [30 30];
% % ��Ŀ �� �κ� ����
% fpath = fullfile(cd,'DB_v2','DB_markset','mark_nose');
% load(fpath);
    
% EMG/marker delay ����
N_delay = 5;

Idx_use_mark_type = 1:3;
Idx_use_emg_feat = 1:4;

% validation ����
val_subject_indepe = 0;
use_saved_network = 0;
% seq size
seq_size = 20;
for i_mark = 12
    if(i_mark==2)
        continue;
    end

    % �ʿ��� feature(ä��)�� ���!
    fpath = 'E:\OneDrive_Hanyang\����\EMG_maker_regression\�ڵ�\DB_v2\DB_markset_10Hz_basecorr_norm_0-1\mark_12';
    load(fpath);
    marker_set = cellfun(@(x) x(:,Idx_use_mark_type),marker_set,'UniformOutput', false); % 6��ä�θ� regression
    fpath = 'E:\OneDrive_Hanyang\����\EMG_maker_regression\�ڵ�\DB_v2\emg_feat_set_10Hz\EMG_feat_normalized';
    load(fpath);
    % emg marker �ð� ����
    feat = cellfun(@(x) x(:,Idx_use_emg_feat),feat,'UniformOutput', false); % 1:4�� ä�θ�
    
    % Seq size�� �ڸ�
    for i=1:numel(feat)
        temp_div = length(marker_set{i})/seq_size;
        temp_z = floor(temp_div);
        if (temp_div -temp_z) ==0
           continue; 
        else
            marker_set{i} = marker_set{i}(1:temp_z*seq_size,:);
            feat{i} = feat{i}(1:temp_z*seq_size,:);
        end
    end
    
    %%  subject dependnt, independet ���Ŀ� ����Train/Test DB ���� 
    % (val_subject_indepe=1 --> subject indep)
    if val_subject_indepe==1
        Xtr_ = feat(Idx_sub4train,:);
        Ttr_ = marker_set(Idx_sub4train,:);
        Idx_sub4test = find(countmember(Idx_sub,Idx_sub4train)==0);
        Xte_ = feat(Idx_sub4test,:);
        Tte_ = marker_set(Idx_sub4test,:);
    else
        Xtr_ = feat(Idx_sub_4testing,Idx_trl4train);
        Ttr_ = marker_set(Idx_sub_4testing,Idx_trl4train);
        Idx_trl4test = find(countmember(Idx_trl,Idx_trl4train)==0);
        Xte_ = feat(Idx_sub_4testing,Idx_trl4test);
        Tte_ = marker_set(Idx_sub_4testing,Idx_trl4test);
    end
        
    %% input and targets of Train
    train_inputs = Xtr_;
    for i= 1:numel(train_inputs)
        train_inputs{i} = mat2cell(Xtr_{i},repmat(seq_size,[length(train_inputs{i})/seq_size,1]),length(Idx_use_emg_feat));  
    end
    train_inputs = cat(1,train_inputs{:});
    train_inputs = cellfun(@(x) x',train_inputs,'UniformOutput', false);
    train_inputs = permute(cat(3,train_inputs{:}),[3 2 1]);
    DB{1} = train_inputs;
    
    train_targets = Ttr_;
    for i= 1:numel(train_targets)
        train_targets{i} = mat2cell(train_targets{i},repmat(seq_size,[length(train_targets{i})/seq_size,1]),3);  
    end
    train_targets = cat(1,train_targets{:});
    train_targets = cellfun(@(x) x',train_targets,'UniformOutput', false);
    train_targets = permute(cat(3,train_targets{:}),[3 2 1]);
    DB{2} = train_targets;
    
    %% input and targets of Test
    test_inputs = Xte_;
    for i= 1:numel(test_inputs)
        test_inputs{i} = mat2cell(Xte_{i},repmat(seq_size,[length(test_inputs{i})/seq_size,1]),length(Idx_use_emg_feat));  
    end
    test_inputs = cat(1,test_inputs{:});
    test_inputs = cellfun(@(x) x',test_inputs,'UniformOutput', false);
    test_inputs = permute(cat(3,test_inputs{:}),[3 2 1]);
    DB{3} = test_inputs;
    
    test_targets = Tte_;
    for i= 1:numel(test_targets)
        test_targets{i} = mat2cell(test_targets{i},repmat(seq_size,[length(test_targets{i})/seq_size,1]),3);  
    end
    test_targets = cat(1,test_targets{:});
    test_targets = cellfun(@(x) x',test_targets,'UniformOutput', false);
    test_targets = permute(cat(3,test_targets{:}),[3 2 1]);
    DB{4} = test_targets;
    save_path = 'C:\Users\CHA\Data\MarkerRegression';
    save(fullfile(save_path,'EMG_Marker_DB.mat'),'DB')
    
 
end
