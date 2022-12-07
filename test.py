from shrinkbench.experiment import PruningExperiment

for  c in [3]:
    for strategy in ['LayerGradCAMShift','LayerSmoothGrad','LayerMagWeight']:
        exp = PruningExperiment(dataset='SST2DATA', 
                                model='BertNet',
                                strategy=strategy,
                                compression=c,
                                train_kwargs={'epochs':1},
                                is_LTH=True)
        exp.run()
