from shrinkbench.experiment import PruningExperiment

for  c in [3.5,3,4,2]:
    for strategy in ['LayerGradCAMShift','LayerSmoothGradCAM','LayerSmoothGrad','LayerSmoothGradCAMShift']:
        if strategy == 'LayerGradCAMShift' and c!=3.5:
            continue
        exp = PruningExperiment(dataset='STSBDATA', 
                                model='BertModelSTSB',
                                strategy=strategy,
                                compression=c,
                                train_kwargs={'epochs':1},
                                is_LTH=True)
        exp.run()
