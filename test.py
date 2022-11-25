from shrinkbench.experiment import PruningExperiment


for strategy in ['LayerMagWeight','GlobalMagWeight']:
    for  c in [1,1.25,1.5,2,2.5]:
        exp = PruningExperiment(dataset='SST2DATA', 
                                model='BertNet',
                                strategy=strategy,
                                compression=c,
                                train_kwargs={'epochs':6})
        exp.run()
