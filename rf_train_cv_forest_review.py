#%%  运行的是区域所有像元的数据点，一个区域一个模型
# 读取库
import numpy as np
import os 
from scipy.io import loadmat
from scipy.stats import pearsonr
import scipy.io as sio
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from itertools import combinations 
import joblib
from sklearn.model_selection import KFold
#Perforing grid search
#
#dir_file = r'/data00/1/djw/China_fire/preprocess_fire_temperature/subregions_tem_gt_5_0.5/pixel_subregions/new_daily_subregions/region4/regress_sample/sample_fire_grass_lon_lat_0.5.mat'
#save_file = r'/data00/1/djw/China_fire/preprocess_fire_temperature/subregions_tem_gt_5_0.5/pixel_subregions/new_daily_subregions/region4/save_model/grass'

dir_file = r'E:\monitor_fire_data\random_tem_gt_5_forest_0.25\regress_sample\split_year_data'
save_file = r'E:\monitor_fire_data\random_tem_gt_5_forest_0.25\save_model\all_models'
save_test_file = r'E:\monitor_fire_data\random_tem_gt_5_forest_0.25\save_model\test_models'

# 获取文件夹内所有 .mat 文件
mat_files = [f for f in os.listdir(dir_file) if f.endswith('.mat')]
#mat_files0 = mat_files[0]
test_rf = np.empty([28,5],dtype=float) # 创建空矩阵
test_vim_fc_rf = np.empty([28,8],dtype=float) # 创建空矩阵
test_vim_frp_rf = np.empty([28,8],dtype=float) # 创建空矩阵
n = 0
# 遍历并读取每个文件
for file in mat_files:
	n = n + 1
	file_path = os.path.join(dir_file, file)
	train =  loadmat(file_path)['train_data'] 
	test =  loadmat(file_path)['test_data'] 
	arr_pc = [2,3,4,6,7,8,11,12] # 26-75 
	X_train = train[:,arr_pc]; X_test = test[:,arr_pc] 
	y_train = train[:,13:15]; y_test = test[:,13:15] 
	# 提前读取相关变量
	lat = test[:,20]; lon = test[:,19] 
	times_test= test[:,1]; times_train= train[:,1]
	col_fc = 0; col_frp = 1
	#=================================================fwi edvi   ndvi =======================================================
	print('==========Num===sample',n,'==of=models=training fwis edvis  ndvi==========================') 
	#arr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23,24,25] # lat lon dis moy  
	#arr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]  # lat 17 lon 18 moy 22  pca 23 
	# 训练集，并转换为2维矩阵     
	TR_X_train = np.array(X_train[:,:]) 
	#TR_X_train = add_noise(TR_X_train)
	TR_Y_fc = np.array(np.log(y_train[:,col_fc])).reshape(np.array(y_train[:,col_fc]).shape[0],-1); ori_TR_Y_fc =  (TR_Y_fc)
	TR_Y_frp = np.array(np.log(y_train[:,col_frp]).reshape(np.array(y_train[:,col_frp]).shape[0],-1)) ; ori_TR_Y_frp =  (TR_Y_frp)
	# 标准化
	TR_X_train = (TR_X_train-np.nanmean(TR_X_train,axis=0))/np.nanstd(TR_X_train,axis=0)

	# 测试集，并转换为2维矩阵
	TE_X_train = np.array(X_test[:,:])
	#TE_X_train = add_noise(TE_X_train)
	TE_Y_fc = np.array(np.log(y_test[:,col_fc])).reshape(np.array(y_test[:,col_fc]).shape[0],-1); ori_TE_Y_fc =  (TE_Y_fc)
	TE_Y_frp = np.array(np.log(y_test[:,col_frp])).reshape(np.array(y_test[:,col_frp]).shape[0],-1); ori_TE_Y_frp =  (TE_Y_frp)
	# 标准化
	TE_X_train = (TE_X_train-np.nanmean(TE_X_train,axis=0))/np.nanstd(TE_X_train,axis=0)
	print('===========rf start======================================')
	#% ==================model 模型  =====================================
	# 模型最优参数化选择 使用frp的mse做判断
	test_edvi_rf1_fc = RandomForestRegressor(n_jobs= -1)
	test_edvi_rf1_frp = RandomForestRegressor(n_jobs= -1)
	param_grid = {'n_estimators': [500], # 400,500 树节点  range(100,1000,50) 
						'bootstrap':[True],
						'max_depth':[60], # 60 70
						'min_samples_leaf':[10] # 10, 20 range(10,10,100) ,30,40,50,60,70,80,90,100 
			}

	rf_edvi_grid_fc = model_selection.GridSearchCV(estimator=test_edvi_rf1_fc, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3, verbose=0)
	rf_edvi_grid_frp = model_selection.GridSearchCV(estimator=test_edvi_rf1_frp, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3, verbose=0)
	# 
	rf_edvi_grid_fc.fit(TR_X_train, TR_Y_fc)
	rf_edvi_grid_frp.fit(TR_X_train, TR_Y_frp)
	# 迭代ParameterGrid对象
	best_params_fc = rf_edvi_grid_fc.best_params_
	best_params_frp = rf_edvi_grid_frp.best_params_

	# 读取最优参数化方案，输入到正式的模型中
	# RF
	# 正式训练xgb 最优的参数化方案为 params
	best_edvi_rf_fc = rf_edvi_grid_fc.best_estimator_
	best_edvi_rf_frp = rf_edvi_grid_frp.best_estimator_
	best_edvi_rf_fc.fit(TR_X_train, TR_Y_fc)
	best_edvi_rf_frp.fit(TR_X_train, TR_Y_frp)

	# 对测试集进行预测
	y_edvi_rf_pred_fc = best_edvi_rf_fc.predict(TE_X_train)
	y_edvi_rf_pred_frp = best_edvi_rf_frp.predict(TE_X_train)
	#还原原来的数据集 并转换为2维

	y_edvi_rf_fc = y_edvi_rf_pred_fc.reshape(np.array(y_edvi_rf_pred_fc).shape[0],-1)#*TE_Y_fc_std+TE_Y_fc_mean
	y_edvi_rf_frp = y_edvi_rf_pred_frp.reshape(np.array(y_edvi_rf_pred_frp).shape[0],-1)#*TE_Y_frp_std+TE_Y_frp_mean
	# 求解统计指标和变量重要性
	r_edvi_rf_frp,p_edvi_rf_frp = pearsonr(ori_TE_Y_frp[:,0],(y_edvi_rf_frp[:,0]))
	r_edvi_rf_fc,p_edvi_rf_fc = pearsonr(ori_TE_Y_fc[:,0],(y_edvi_rf_fc[:,0]))
	print('r_edvi_rf_fc',r_edvi_rf_fc,'r_edvi_rf_frp',r_edvi_rf_frp)
	mse_edvi_rf_frp = mean_squared_error(ori_TE_Y_frp, (y_edvi_rf_frp))
	mse_edvi_rf_fc = mean_squared_error(ori_TE_Y_fc, (y_edvi_rf_fc))
	nrmse_edvi_fc =  mean_squared_error(ori_TE_Y_fc, (y_edvi_rf_fc), squared=False)/np.nanmean((ori_TE_Y_fc))
	nrmse_edvi_frp =  mean_squared_error(ori_TE_Y_frp, (y_edvi_rf_frp), squared=False)/np.nanmean((ori_TE_Y_frp))
	vim_edvi_rf_fc = best_edvi_rf_fc.feature_importances_
	vim_edvi_rf_frp = best_edvi_rf_frp.feature_importances_
	test_vim_fc_rf[n-1,:] = np.array([vim_edvi_rf_fc])
	test_vim_frp_rf[n-1,:] = np.array([vim_edvi_rf_frp])
	print('r_edvi_vim_fc',vim_edvi_rf_fc,'r_edvi_vim_frp',vim_edvi_rf_frp)
	# 输出训练号的所有参数|
	dic_rf_data = {
		'times_test':times_test,'times_train':times_train,'test':X_test[:,:],
		'sample':np.array([(y_edvi_rf_fc[:,0]),(y_edvi_rf_frp[:,0]),(ori_TE_Y_fc[:,0]),(ori_TE_Y_frp[:,0])]).astype(float),
		'vim_rf':np.array([(vim_edvi_rf_fc),(vim_edvi_rf_frp)]).astype(float),
		'metrics':np.array([(r_edvi_rf_fc),p_edvi_rf_fc,r_edvi_rf_frp,p_edvi_rf_frp,mse_edvi_rf_fc,mse_edvi_rf_frp,
							nrmse_edvi_fc,nrmse_edvi_frp]).astype(float),
							'cord':np.array([(lat),(lon)]).astype(float),
							}
	dic_edvi_rf_model  ={'rf_fc':best_edvi_rf_fc,'rf_frp':best_edvi_rf_frp}
	dic_edvi_rf_params  ={'parameters_fc':best_params_fc,'parameters_frp':best_params_frp}
	file_edvi_data_rf =  'RF_edvi_data'+'_g_'+str(n)+'.mat'
	model_edvi_rf =  'RF_fwis_edvi_g_'+str(n)+'.joblib'
	savedirpath_model_edvi_rf =  os.path.join(save_file,model_edvi_rf) 
	joblib.dump(dic_edvi_rf_model, savedirpath_model_edvi_rf)

	savedirpath_edvi_rf =  os.path.join(save_file,file_edvi_data_rf)
	sio.savemat(savedirpath_edvi_rf,dic_rf_data) #保存为mat文件 dic_multi_data
	file_edvi_params_rf = 'RF_edvi_parames_g_'+str(n)+'.mat'
	savedirpath_edvi_rf =  os.path.join(save_file,file_edvi_params_rf)
	sio.savemat(savedirpath_edvi_rf,dic_edvi_rf_params) #保存为mat文件 dic_multi_data

	print('==========Num===sample',n,'==of=models=training fwis ==========================') 
	#arr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,21,22,23,24,25] # lat lon dis moy  
	arr = [3,4,5,6,7]  # lat 17 lon 18 moy 22  pca 23 
	# 训练集，并转换为2维矩阵     
	TR_X_train = np.array(X_train[:,arr]) 
	#TR_X_train = add_noise(TR_X_train)
	TR_Y_fc = np.array(np.log(y_train[:,col_fc])).reshape(np.array(y_train[:,col_fc]).shape[0],-1); ori_TR_Y_fc =  (TR_Y_fc)
	TR_Y_frp = np.array(np.log(y_train[:,col_frp]).reshape(np.array(y_train[:,col_frp]).shape[0],-1)) ; ori_TR_Y_frp =  (TR_Y_frp)
	# 标准化
	TR_X_train = (TR_X_train-np.nanmean(TR_X_train,axis=0))/np.nanstd(TR_X_train,axis=0)

	# 测试集，并转换为2维矩阵
	TE_X_train = np.array(X_test[:,arr])
	#TE_X_train = add_noise(TE_X_train)
	TE_Y_fc = np.array(np.log(y_test[:,col_fc])).reshape(np.array(y_test[:,col_fc]).shape[0],-1); ori_TE_Y_fc =  (TE_Y_fc)
	TE_Y_frp = np.array(np.log(y_test[:,col_frp])).reshape(np.array(y_test[:,col_frp]).shape[0],-1); ori_TE_Y_frp =  (TE_Y_frp)
	# 标准化
	TE_X_train = (TE_X_train-np.nanmean(TE_X_train,axis=0))/np.nanstd(TE_X_train,axis=0)
	print('===========rf start======================================')
	#% ==================model 模型  =====================================
	# 模型最优参数化选择 使用frp的mse做判断
	test_rf1_fc = RandomForestRegressor(n_jobs= -1)
	test_rf1_frp = RandomForestRegressor(n_jobs= -1)
	param_grid = {'n_estimators': [500], # 400,500 树节点  range(100,1000,50) 
						'bootstrap':[True],
						'max_depth':[60], # 60 70
						'min_samples_leaf':[10] # 10, 20 range(10,10,100) ,30,40,50,60,70,80,90,100 
			}

	rf_grid_fc = model_selection.GridSearchCV(estimator=test_rf1_fc, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3, verbose=0)
	rf_grid_frp = model_selection.GridSearchCV(estimator=test_rf1_frp, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=3, verbose=0)
	# 
	rf_grid_fc.fit(TR_X_train, TR_Y_fc)
	rf_grid_frp.fit(TR_X_train, TR_Y_frp)
	# 迭代ParameterGrid对象
	best_params_fc = rf_grid_fc.best_params_
	best_params_frp = rf_grid_frp.best_params_

	# 读取最优参数化方案，输入到正式的模型中
	# RF
	# 正式训练xgb 最优的参数化方案为 params
	best_rf_fc = rf_grid_fc.best_estimator_
	best_rf_frp = rf_grid_frp.best_estimator_
	best_rf_fc.fit(TR_X_train, TR_Y_fc)
	best_rf_frp.fit(TR_X_train, TR_Y_frp)

	# 对测试集进行预测
	y_rf_pred_fc = best_rf_fc.predict(TE_X_train)
	y_rf_pred_frp = best_rf_frp.predict(TE_X_train)
	#还原原来的数据集 并转换为2维

	y_rf_fc = y_rf_pred_fc.reshape(np.array(y_rf_pred_fc).shape[0],-1)#*TE_Y_fc_std+TE_Y_fc_mean
	y_rf_frp = y_rf_pred_frp.reshape(np.array(y_rf_pred_frp).shape[0],-1)#*TE_Y_frp_std+TE_Y_frp_mean
	# 求解统计指标和变量重要性
	r_rf_frp,p_rf_frp = pearsonr(ori_TE_Y_frp[:,0],(y_rf_frp[:,0]))
	r_rf_fc,p_rf_fc = pearsonr(ori_TE_Y_fc[:,0],(y_rf_fc[:,0]))
	print('r_rf_fc',r_rf_fc,'r_rf_frp',r_rf_frp)
	mse_rf_frp = mean_squared_error(ori_TE_Y_frp, (y_rf_frp))
	mse_rf_fc = mean_squared_error(ori_TE_Y_fc, (y_rf_fc))
	nrmse_fc =  mean_squared_error(ori_TE_Y_fc, (y_rf_fc), squared=False)/np.nanmean((ori_TE_Y_fc))
	nrmse_frp =  mean_squared_error(ori_TE_Y_frp, (y_rf_frp), squared=False)/np.nanmean((ori_TE_Y_frp))
	vim_rf_fc = best_rf_fc.feature_importances_
	vim_rf_frp = best_rf_frp.feature_importances_
	test_rf[n-1,:] = np.array([n,r_rf_fc,r_rf_frp,r_edvi_rf_fc,r_edvi_rf_frp])
	
	# 输出训练号的所有参数|
	dic_rf_data = {
		'times_test':times_test,'times_train':times_train,'test':X_test[:,arr],
		'sample':np.array([(y_rf_fc[:,0]),(y_rf_frp[:,0]),(ori_TE_Y_fc[:,0]),(ori_TE_Y_frp[:,0])]).astype(float),
		'vim_rf':np.array([(vim_rf_fc),(vim_rf_frp)]).astype(float),
		'metrics':np.array([(r_rf_fc),p_rf_fc,r_rf_frp,p_rf_frp,mse_rf_fc,mse_rf_frp,
							nrmse_fc,nrmse_frp]).astype(float),
							'cord':np.array([(lat),(lon)]).astype(float),
							}
	dic_rf_model  ={'rf_fc':best_rf_fc,'rf_frp':best_rf_frp}
	dic_rf_params  ={'parameters_fc':best_params_fc,'parameters_frp':best_params_frp}
	file_data_rf =  'RF_fwis_data'+'_g_'+str(n)+'.mat'
	model_rf =  'RF_fwis_g_'+str(n)+'.joblib'
	savedirpath_model_rf =  os.path.join(save_file,model_rf) 
	joblib.dump(dic_rf_model, savedirpath_model_rf)

	savedirpath_rf =  os.path.join(save_file,file_data_rf)
	sio.savemat(savedirpath_rf,dic_rf_data) #保存为mat文件 dic_multi_data
	file_params_rf = 'RF_parames_fwis_g_'+str(n)+'.mat'
	savedirpath_rf =  os.path.join(save_file,file_params_rf)
	sio.savemat(savedirpath_rf,dic_rf_params) #保存为mat文件 dic_multi_data
	print('===========Fwis-ndvi-edvi==随机森林 finish======================================')

	#	sio.savemat(savedirpath,dic_multi_data) #保存为mat文件 dic_multi_data
#% 保存所有的测试结果
dic_models_test = {
'rf': test_rf.astype(float),
'vim_fc':test_vim_fc_rf.astype(float),
'vim_frp':test_vim_frp_rf.astype(float)}
filename_test = 'test_data.mat'
save_test_filepath = os.path.join(save_test_file ,filename_test) 	
sio.savemat(save_test_filepath,dic_models_test) #保存为mat文件 dic_multi_data
	#sio.savemat(save_test_filepath,dic_xgb_test) #保存为mat文件 dic_multi_data
	#print('没有值',subdir[j])      

# %%
