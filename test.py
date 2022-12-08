from shrinkbench.experiment import PruningExperiment

for  c in [2,4]:
    for strategy in ['LayerGradCAMShift','LayerSmoothGrad','LayerMagWeight','LayerSmoothGradCAMShift']:
        if c==3 and strategy!='LayerSmoothGradCAMShift':
            continue
        exp = PruningExperiment(dataset='COLADATA', 
                                model='BertNet',
                                strategy=strategy,
                                compression=c,
                                train_kwargs={'epochs':1},
                                is_LTH=True)
        exp.run()