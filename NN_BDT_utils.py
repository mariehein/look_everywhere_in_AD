import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path
from tensorflow import keras
models = keras.models
layers = keras.layers
regularizers = keras.regularizers

def to_categorical(Y, N_classes=2):
	Y=np.array(Y,dtype=int)
	return np.eye(N_classes)[Y]

def make_one_array(twod_arr,new_arr):
    '''Helper function to save ROC curves'''
    cols = twod_arr.shape[1]
    new_len = len(new_arr)
    if new_len < cols: 
        new_arr = np.pad(new_arr, (0, cols-new_len))
    elif new_len > cols: 
        twod_arr = np.pad(twod_arr, ((0,0), (0,new_len-cols)))
    return np.vstack([twod_arr, new_arr])

def plot_roc(test_results, test_labels, directory, title="roc"):
    '''Helper function saving ROC curves from multiple runs into one 2D array'''
    fpr, tpr, _ = roc_curve(test_labels, test_results)

    fpr_path = directory+"fpr_"+title+".npy"
    tpr_path = directory+"tpr_"+title+".npy"

    if Path(tpr_path).is_file():
        np.save(tpr_path, make_one_array(np.load(tpr_path), tpr))
        np.save(fpr_path, make_one_array(np.load(fpr_path), fpr))
    else: 
        np.save(tpr_path,np.array([tpr]))
        np.save(fpr_path,np.array([fpr]))
	
    return roc_auc_score(test_labels, test_results)

def make_NN_model(activation="relu",hidden=3,inputs=4,lr=1e-3,dropout=0.1, l1=0, l2 =0, momentum = 0.9, label_smoothing=0):
	model = models.Sequential()
	model.add(layers.Dense(64,input_shape=(inputs,)))
	for i in range(hidden-1):
		if activation =="relu":
			model.add(layers.ReLU())
		elif activation == "leaky":
			model.add(layers.LeakyReLU(alpha=0.1))
		model.add(layers.Dropout(dropout))
		model.add(layers.Dense(64,kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
		model.add(layers.ReLU())
	model.add(layers.Dense(2, activation="softmax"))

	loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

	model.compile(
		loss=loss,
		optimizer=keras.optimizers.Adam(lr, beta_1=momentum),
		metrics=["accuracy"],
	)
	return model

def NN_model_training(X_train, Y_train, noearlystopping, revert=True, inputs=4):
	model = make_NN_model(inputs=inputs)
	if noearlystopping:
		results = model.fit(
				X_train,
				Y_train,
				batch_size=1024,
				epochs=100,
				shuffle=True,
				verbose=2,
			)
	else: 
		callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min', restore_best_weights=revert)]
		val_frac = 0.5
		results = model.fit(
				X_train,
				Y_train,
				batch_size=1024,
				epochs=100,
				shuffle=True,
				verbose=2,
				validation_split=val_frac,
				callbacks=callbacks,
			)
	return model

def NN_training_and_preds(args, X_train, Y_train, get_preds_list, inputs=4):
	Y_train = to_categorical(Y_train)
	model = NN_model_training(X_train, Y_train, args.noearlystopping, args.revert, inputs=inputs)
	preds_list = [np.zeros(arr.shape[0], dtype=float) for arr in get_preds_list]

	for i,arr in enumerate(get_preds_list):
		preds_list[i]=model.predict(arr)[:,1]
	return preds_list


def BDT_training_and_preds(args, X_train, Y_train, get_preds_list, run):
	from sklearn.ensemble import HistGradientBoostingClassifier
	preds_list = [np.zeros(arr.shape[0], dtype=float) for arr in get_preds_list]
	for j in range(args.ensemble):
		np.random.seed(args.ensemble*run+j)
		model = HistGradientBoostingClassifier(early_stopping=not args.noearlystopping, validation_fraction=0.5)
		results = model.fit(X_train, Y_train)

		if args.revert:
			min_ind = max(np.argmin(results.validation_score_)-1,0)
			for i,arr in enumerate(get_preds_list):
				for l, v in enumerate(model.staged_predict_proba(arr)):
					if l==min_ind:
						preds_list[i] += v[:,1]/args.ensemble
		else:
			for i,arr in enumerate(get_preds_list):
				preds_list[i] += model.predict_proba(arr)[:,1]/args.ensemble

	return preds_list