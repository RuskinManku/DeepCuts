from shrinkbench.experiment import PruningExperiment


for strategy in ['LayerGradCAM']:
    for  c in [1.5]:
        exp = PruningExperiment(dataset='SST2DATA', 
                                model='BertNet',
                                strategy=strategy,
                                compression=c,
                                train_kwargs={'epochs':1},
                                is_LTH=False)
        exp.run()
