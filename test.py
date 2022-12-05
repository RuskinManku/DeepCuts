from shrinkbench.experiment import PruningExperiment

for  c in [3.5,3,4,2]:
    for strategy in ['LayerGradCAMShift','LayerSmoothGradCAM','LayerSmoothGrad','LayerSmoothGradCAMShift']:
        if strategy == 'LayerGradCAMShift' and c!=3.5:
            continue
        try:
            exp = PruningExperiment(dataset='SST2DATA', 
                                    model='BertNet',
                                    strategy=strategy,
                                    compression=c,
                                    train_kwargs={'epochs':1},
                                    is_LTH=True)
            exp.run()
        except:
            print("FAILED FOR: {}".format(strategy + '_', c))
